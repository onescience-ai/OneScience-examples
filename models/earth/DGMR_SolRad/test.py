#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DGMR_SolRad 适配 / 测试脚本
适用环境: SCNET 超算  flagos_earth_onecode 镜像 / Hygon DCU K100AI

设计原则: 前向计算与官方 inference.py **完全一致**
(相同的 Predictor、相同的 5 个输入键、相同的 clearsky 换算、相同的输出形状),
仅额外增加下面 4 项用于"适配 + 出测试报告":
  1) FlagGems(FlagOS)算子库的启用,以及"某算子不支持时回退到原生 torch"的开关;
  2) 推理耗时统计(测试报告的性能指标);
  3) 输出张量的 形状 / 数值范围 / NaN,Inf 校验;
  4) 若样本 npz 中带真值(ground truth)则自动计算 MAE/RMSE/Bias/R2,否则明确跳过。

用法:
  cd /root/private_data/DGMR_SolRad
  python test.py --model-type DGMR_SO --basetime 202504131100

可选环境变量:
  USE_FLAGGEMS=1   是否启用 FlagGems(默认 1;设 0 则用 DCU 原生 torch 作基线对照)
  SEED=0           随机种子(可选)。DGMR_SO 内部对隐变量做高斯采样,输出本身是随机的;
                   设定 SEED 可让多次运行结果可复现。Generator_only 无采样,天然确定。
"""

import os
import time
import argparse
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1) FlagGems (FlagOS) 算子库启用
#    - flagos 镜像通过把 Triton 算子注册进 PyTorch 的 ATen 后端来加速/适配。
#    - 默认启用全部 FlagGems 算子。若日志报告某算子不支持或精度异常,
#      把该算子名加入 FLAGGEMS_UNUSED 列表 -> 该算子回退原生 torch,其余仍走 FlagGems。
#    - 算子名映射表见:
#      https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/__init__.py
# ---------------------------------------------------------------------------
USE_FLAGGEMS = os.environ.get("USE_FLAGGEMS", "1") == "1"

# >>> 下列算子已由 DCU 实测确认需回退,保持原样即可。 <<<
# 若后续换平台/换镜像又报 "xxx is not supported / not implemented in flag_gems",
# 把报错点名的算子加进来再重跑(该算子回退原生 torch,其余仍走 FlagGems)。
# 其它可能需要回退的候选(按报错提示挑对应项填入):
#   "upsample_nearest2d" -- Up_GBlock 的 nn.Upsample(mode="nearest")
#   "bmm" / "mm"         -- 仅 DGMR_SO:注意力层 torch.einsum 会下沉到这些矩阵乘
FLAGGEMS_UNUSED = [
    # 以下为 DCU K500SM_AI + flagos_earth_onecode 实测确认需回退的算子:
    "pixel_unshuffle",   # FlagGems 版仅支持4D,本模型输入5D
    "pixel_shuffle",     # 同上,预防性回退
    "reflection_pad2d",  # ConvGRU 的 reflect padding
    "mv",                # spectral_norm 功率迭代;Triton autotune 开销极大
    "dot",               # 同上
]


def try_enable_flaggems():
    """尝试启用 FlagGems。返回是否成功启用。任何失败都不阻断,退回原生 torch。"""
    if not USE_FLAGGEMS:
        print("[FlagGems] USE_FLAGGEMS=0 -> 使用 DCU 原生 torch(不启用 FlagGems,作基线对照)")
        return False
    try:
        import flag_gems
    except Exception as e:
        print(f"[FlagGems] 未安装/导入失败({e}) -> 使用 DCU 原生 torch 继续运行")
        return False
    try:
        # 平台文档给出的参数: unused / record / path / once
        flag_gems.enable(
            unused=FLAGGEMS_UNUSED,
            record=True,
            path="./flaggems_ops.log",
            once=True,
        )
    except TypeError:
        # 兼容不同版本的 enable() 签名: 退化为最简调用
        flag_gems.enable(unused=FLAGGEMS_UNUSED)
    print(f"[FlagGems] 已启用 (unused={FLAGGEMS_UNUSED});算子调用日志 -> ./flaggems_ops.log")
    return True


def data_loading(BASETIME, device):
    npz_path = f"./sample_data/sample_{BASETIME}.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"找不到样本数据 {npz_path} 。\n"
            f">> 多半是 Git LFS 大文件没拉下来。请在有网环境执行  git lfs pull  "
            f"或  hf download ,并用  du -sh {npz_path}  确认它是真实二进制而非几百字节的指针。"
        )
    data_npz = np.load(npz_path)
    inputs = {}
    for key in data_npz:
        inputs[key] = torch.from_numpy(data_npz[key]).to(device)
    print(f"[数据] 载入 {npz_path};包含键: {list(data_npz.keys())}")
    for k, v in inputs.items():
        print(f"        - {k:10s} shape={tuple(v.shape)} dtype={v.dtype}")
    return inputs


def model_loading(model_type, device):
    if model_type == "DGMR_SO":
        ckpt_path = "./model_weights/DGMR_SO/ft36/weights.ckpt"
    elif model_type == "Generator_only":
        ckpt_path = "./model_weights/Generator_only/ft36/weights.ckpt"
    else:
        raise ValueError(f"未知 model_type: {model_type}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到权重 {ckpt_path} 。\n"
            f">> 多半是 Git LFS 大文件没拉下来。请在有网环境执行  git lfs pull  或  hf download 。"
        )
    # 与官方一致: 从 model_architect 载入 Predictor(该目录随仓库一起下载)
    from model_architect.inference_model import Predictor

    model = Predictor(model_type=model_type)
    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt["generator_state_dict"])
    model.eval()
    model.to(device)
    print(f"[模型] {model_type} 权重载入成功: {ckpt_path}")
    return model


def compute_optional_metrics(pred_srad, inputs):
    """若样本里带真值就算误差指标;没有则返回 None(生成式预测,样本通常不含未来真值)。"""
    gt_key = None
    for cand in ["target", "label", "gt", "truth", "srad_true", "y_true", "obs", "y"]:
        if cand in inputs:
            gt_key = cand
            break
    if gt_key is None:
        return None
    y_true = inputs[gt_key].detach().cpu().numpy().reshape(-1).astype(np.float64)
    y_pred = pred_srad.detach().cpu().numpy().reshape(-1).astype(np.float64)
    n = min(y_true.size, y_pred.size)
    y_true, y_pred = y_true[:n], y_pred[:n]
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"key": gt_key, "MAE": mae, "RMSE": rmse, "Bias": bias, "R2": r2}


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="DGMR_SO",
                        choices=["Generator_only", "DGMR_SO"])
    parser.add_argument("--basetime", type=str, default="202504131100")
    parser.add_argument("--pred-step", type=int, default=36)
    return parser.parse_args()


def main():
    args = arg_parse()

    # 可选随机种子: DGMR_SO 内部对隐变量做高斯采样 -> 输出是随机样本;
    # 设定 SEED 让多次运行可复现(Generator_only 无采样,不受影响)。
    seed_env = os.environ.get("SEED", "")
    seed = int(seed_env) if seed_env.strip() != "" else None
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print("=" * 74)
    print(" DGMR_SolRad 适配测试")
    print("=" * 74)
    if seed is not None:
        print(f" 随机种子 SEED     : {seed}")
    print(f" torch 版本        : {torch.__version__}")
    print(f" CUDA/DCU 可用     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f" 设备名称          : {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    try_enable_flaggems()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 运行设备          : {device}")
    print("-" * 74)

    inputs = data_loading(args.basetime, device)
    model = model_loading(args.model_type, device)

    # 预热一次(不计时): 消除首次 kernel 编译/加载开销,使耗时更可比
    with torch.no_grad():
        _ = model(inputs["Himawari"], inputs["WRF"], inputs["topo"],
                  inputs["time_feat"], pred_step=args.pred_step)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 正式推理 + 计时
    t0 = time.time()
    with torch.no_grad():
        pred_clr_idx = model(inputs["Himawari"], inputs["WRF"], inputs["topo"],
                             inputs["time_feat"], pred_step=args.pred_step)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_sec = time.time() - t0

    # 后处理: clearsky index -> solar radiation(与官方 inference.py 完全一致)
    pred_clr_idx = pred_clr_idx.squeeze(2).clamp(0, 1)
    pred_srad = pred_clr_idx * inputs["clearsky"]        # dim: (1, 36, 512, 512)

    out_path = f"./pred_{args.basetime}_{args.model_type}.npy"
    np.save(out_path, pred_srad.cpu().numpy())

    # ---- 校验与指标 ----
    arr = pred_srad.detach().cpu().numpy()
    has_bad = bool(np.isnan(arr).any() or np.isinf(arr).any())
    print("-" * 74)
    print(" [结果] 前向推理成功")
    print(f"   输出文件        : {out_path}")
    print(f"   输出形状        : {tuple(arr.shape)}   (期望 (1, {args.pred_step}, 512, 512))")
    print(f"   数值范围        : min={float(np.nanmin(arr)):.4f}  "
          f"max={float(np.nanmax(arr)):.4f}  mean={float(np.nanmean(arr)):.4f}")
    print(f"   含 NaN/Inf      : {has_bad}")
    print(f"   推理耗时        : {infer_sec * 1000:.1f} ms   ({infer_sec:.3f} s)")

    m = compute_optional_metrics(pred_srad, inputs)
    if m is None:
        print("   误差指标        : 样本未包含真值(生成式预测),不计算 MAE/RMSE;"
              "以 推理耗时 + 输出校验 作为测试指标")
    else:
        print(f"   误差指标(vs '{m['key']}'): MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"Bias={m['Bias']:+.4f}  R2={m['R2']:.4f}")
    if args.model_type == "DGMR_SO" and seed is None:
        print("   备注            : DGMR_SO 采样随机隐变量,未设 SEED 时每次输出会不同(属正常现象)")
    print("=" * 74)
    print(" 校验结论: " + ("通过 ✓(形状正确且无 NaN/Inf)" if (not has_bad and arr.shape[1] == args.pred_step)
                          else "请检查(形状或数值异常)"))
    print("=" * 74)
    print("Done")


if __name__ == "__main__":
    main()
