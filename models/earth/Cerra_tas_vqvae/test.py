# -*- coding: utf-8 -*-
"""
predictia/cerra_tas_vqvae  适配/复现测试脚本  (v2)
===========================================================
模型类型 : VQ-VAE (diffusers.VQModel) —— 单通道 80x80 温度场(CERRA tas)自编码器
用途     : 在国产 DCU (FlagOS: flagos_earth_onecode) 上验证该模型可加载、可前向推理、
           输出形状正确、无算子报错,并给出重构误差与推理耗时等指标。

v2 相对 v1 的改动(针对 flagos_earth_onecode 镜像实测报错)
-----------------------------------------------------------
镜像里存在一个已损坏的第三方动态库:
    libtensorflow_cc.so.2: undefined symbol: ncclCommRegister
它会在 `import flag_gems` 之后污染进程,并让随后的 `import diffusers` 直接 ImportError。
本版本做了四层处理,按顺序自动尝试,任一层成功即继续:

  A. 先设环境变量(USE_TF=0 / USE_TORCH=1 / USE_JAX=0),让下游库根本不去探测 TensorFlow;
     并在 sys.modules 里放一个 tensorflow 占位模块,阻止那个坏 .so 被真正加载。
  B. **先 import diffusers,后启用 FlagGems**(v1 是反的)。这样 diffusers 的依赖链在
     干净的进程里完成导入。FlagGems 是在 torch dispatch 层全局接管算子的,
     在模型构建/推理之前启用即可生效,不影响其作用。
  C. 若 `from diffusers import VQModel` 仍失败:先只 `import diffusers`(它是惰性加载,
     很轻),把 accelerate / xformers / onnx / transformers 等**可选**依赖的可用性标志
     强制置 False,再导入 VQModel —— 绕开出问题的那条可选依赖链。
  D. 若 diffusers 彻底不可用:回退到同目录的 `vqvae_torch.py` —— 一个与
     diffusers.VQModel 结构等价的纯 PyTorch 实现,以 strict=True 装载官方权重
     (参数名/形状对不上就直接报错,不会蒙混过关)。日志与报告里会明确标注用的是哪条路径。

任何一层失败都会打印**完整 traceback**,方便定位。

其它设计要点:
1. 本模型不是文本条件的 image-to-image 扩散管线。HuggingFace 模型卡里自动生成的
   DiffusionPipeline.from_pretrained / AutoModel.from_pretrained / pipeline("image-to-image")
   对本模型全部无效。正确加载方式是 diffusers.VQModel,本脚本已按正确方式实现。
2. 优先从本地目录加载(超算容器通常无外网)。
3. 所有关键指标都会打印成「报告块」并写入 result.log,方便直接誊抄进测试报告。
   —— 报告里的数值请以本脚本实际运行输出为准,不要手工编造。
"""

import os
import sys
import time
import json
import types
import argparse
import traceback
import importlib
import importlib.machinery

# ======================================================================
# 0-A. 在导入任何重库之前:关掉 TensorFlow / Flax 探测,并挡住那个坏 .so
# ======================================================================
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BLOCK_TF = os.environ.get("BLOCK_TENSORFLOW", "1") == "1"


def _install_tf_stub():
    """把 tensorflow 换成一个无害的占位模块,阻止镜像里那个 undefined symbol 的 .so 被加载。
    本模型完全不需要 TensorFlow,只是某些库在导入时会去探测它。"""
    if "tensorflow" in sys.modules:
        return False

    class _Stub(types.ModuleType):
        __version__ = "0.0.0-stub"
        __path__ = []           # 让它看起来像个包,子模块也能被 import

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            sub = _Stub(f"{self.__name__}.{name}")
            sub.__spec__ = importlib.machinery.ModuleSpec(sub.__name__, None)
            sys.modules[sub.__name__] = sub
            return sub

    m = _Stub("tensorflow")
    m.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
    sys.modules["tensorflow"] = m
    return True


import numpy as np

# ----------------------------------------------------------------------
# 0-B. 命令行参数
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="cerra_tas_vqvae adaptation test")
parser.add_argument("--model_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                    help="包含 config.json 与 diffusion_pytorch_model.bin 的目录;默认脚本所在目录")
parser.add_argument("--repo_id", type=str, default="predictia/cerra_tas_vqvae",
                    help="本地加载失败时,若有外网则回退用该 HF repo id 下载")
parser.add_argument("--runs", type=int, default=50, help="计时的重复推理次数")
parser.add_argument("--warmup", type=int, default=5, help="计时前的预热次数")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--save_fig", action="store_true", help="是否保存输入/重构可视化 png(需 matplotlib)")
parser.add_argument("--force_fallback", action="store_true",
                    help="调试用:跳过 diffusers,直接用纯 PyTorch 等价实现")
args = parser.parse_args()

np.random.seed(args.seed)

# 同时把标准输出复制到 result.log
LOG_PATH = os.path.join(args.model_dir, "result.log")


class _Tee:
    def __init__(self, *streams): self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_logfile = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _logfile)
sys.stderr = _Tee(sys.__stderr__, _logfile)


def hr(title=""):
    print("\n" + "=" * 68)
    if title:
        print(title)
        print("=" * 68)


# ======================================================================
# 1. 导入 torch 与 VQModel(FlagGems 放到后面再启用)
# ======================================================================
hr("[STEP 1] 导入 torch 与 VQ-VAE 模型类")

if BLOCK_TF:
    if _install_tf_stub():
        print("[env] 已用占位模块屏蔽 tensorflow(避免镜像内 libtensorflow_cc.so.2 "
              "undefined symbol: ncclCommRegister 影响导入;本模型不需要 TF)")
        print("      如需关闭该屏蔽:BLOCK_TENSORFLOW=0 bash run.sh")

import torch  # noqa: E402

print(f"[env] python = {sys.version.split()[0]}")
print(f"[env] torch  = {torch.__version__}")
print(f"[env] numpy  = {np.__version__}")

VQModel = None
BACKEND = None            # "diffusers" | "pure-torch-fallback"
DIFFUSERS_VER = "N/A"

_OPTIONAL_FLAGS = [
    # diffusers.utils.import_utils 里的可选依赖开关。全部置 False 后,
    # diffusers 只会走「纯 torch」路径,不再去 import 这些可能有问题的库。
    "_accelerate_available", "_xformers_available", "_onnx_available",
    "_torch_npu_available", "_transformers_available", "_flax_available",
    "_note_seq_available", "_k_diffusion_available", "_bitsandbytes_available",
    "_opencv_available", "_invisible_watermark_available", "_torchvision_available",
    "_peft_available", "_torchsde_available", "_wandb_available",
    "_tensorboard_available", "_compel_available", "_imageio_available",
    "_bs4_available", "_ftfy_available", "_unidecode_available",
    "_sentencepiece_available", "_timm_available", "_safetensors_available",
]


def _strategy_direct():
    from diffusers import VQModel as _M
    return _M


def _strategy_patched_flags():
    """先只 import diffusers(惰性、很轻),关掉所有可选依赖,再导入 VQModel。"""
    import diffusers  # noqa: F401
    iu = sys.modules.get("diffusers.utils.import_utils")
    turned_off = []
    if iu is not None:
        for flag in _OPTIONAL_FLAGS:
            if getattr(iu, flag, None):
                setattr(iu, flag, False)
                turned_off.append(flag.strip("_").replace("_available", ""))
    if turned_off:
        print(f"        已临时关闭可选依赖: {', '.join(turned_off)}")
    last = None
    for modname in ("diffusers.models.autoencoders.vq_model",
                    "diffusers.models.vq_model"):
        try:
            return getattr(importlib.import_module(modname), "VQModel")
        except Exception as e:                       # noqa: BLE001
            last = e
    from diffusers import VQModel as _M              # 最后再试一次常规路径
    if last is not None:
        pass
    return _M


def _strategy_fallback():
    import vqvae_torch
    return vqvae_torch


if args.force_fallback:
    print("[import] --force_fallback:跳过 diffusers,直接使用纯 PyTorch 等价实现")
else:
    for name, fn in (("diffusers 常规导入", _strategy_direct),
                     ("diffusers + 关闭可选依赖", _strategy_patched_flags)):
        try:
            print(f"[import] 尝试:{name} ...")
            VQModel = fn()
            import diffusers
            DIFFUSERS_VER = getattr(diffusers, "__version__", "unknown")
            BACKEND = "diffusers"
            print(f"[import] 成功。diffusers = {DIFFUSERS_VER}")
            break
        except Exception:                            # noqa: BLE001
            print(f"[import] {name} 失败,完整 traceback:")
            print(traceback.format_exc())

if VQModel is None:
    print("[import] diffusers 不可用,回退到纯 PyTorch 等价实现 vqvae_torch.py")
    sys.path.insert(0, args.model_dir)
    try:
        VQModel = _strategy_fallback()
        BACKEND = "pure-torch-fallback"
        DIFFUSERS_VER = "未使用(等价实现)"
        print("[import] 已启用纯 PyTorch 等价实现(将以 strict=True 校验官方权重)")
    except Exception:
        print("[FATAL] 纯 PyTorch 回退实现也无法导入,完整 traceback:")
        print(traceback.format_exc())
        sys.exit(1)

# ======================================================================
# 2. FlagOS / FlagGems 算子加速(在模型构建与推理之前启用)
#    参数含义:unused=禁用列表(某算子不支持时填入以回退 torch),record=记录算子日志,
#             path=日志路径,once=同一算子只记一次。
# ======================================================================
hr("[STEP 2] 初始化 FlagGems 算子加速")
USE_FLAGGEMS = os.environ.get("USE_FLAGGEMS", "1") == "1"
FLAGGEMS_ON = False
if USE_FLAGGEMS:
    try:
        import flag_gems
        try:
            # 若某算子在 DCU 上报错,把算子名加入 unused 即可禁用 FlagGems、回退 torch
            flag_gems.enable(
                unused=[],                                    # 例如 ["cumsum", "argmax"]
                record=True,
                path=os.path.join(args.model_dir, "flaggems_ops.log"),
                once=True,
            )
        except TypeError:
            flag_gems.enable()                                # 兼容不同版本的 enable() 签名
        FLAGGEMS_ON = True
        print("[FlagGems] 已启用(算子日志见 flaggems_ops.log)")
    except Exception as e:                                    # noqa: BLE001
        print(f"[FlagGems] 未启用,回退原生 torch({type(e).__name__}: {e})")
else:
    print("[FlagGems] 环境变量 USE_FLAGGEMS!=1,跳过")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        print(f"[env] device = cuda  ({torch.cuda.get_device_name(0)})")
    except Exception:
        print("[env] device = cuda")
else:
    print("[env] device = cpu  (未检测到 GPU/DCU)")

# ======================================================================
# 3. 加载模型:本地优先,失败再回退 HF repo id(需外网)
# ======================================================================
hr("[STEP 3] 加载 VQ-VAE 模型")
print(f"[load] 后端 = {BACKEND}")
has_local = os.path.exists(os.path.join(args.model_dir, "config.json")) and (
    os.path.exists(os.path.join(args.model_dir, "diffusion_pytorch_model.bin")) or
    os.path.exists(os.path.join(args.model_dir, "diffusion_pytorch_model.safetensors"))
)

t0 = time.time()
if has_local:
    print(f"[load] 从本地目录加载: {args.model_dir}")
    model = VQModel.from_pretrained(args.model_dir)
else:
    print(f"[load] 本地未找到权重,尝试从 HF 下载: {args.repo_id}(需外网)")
    model = VQModel.from_pretrained(args.repo_id)
load_sec = time.time() - t0

model = model.to(DEVICE).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"[load] 加载耗时  : {load_sec:.3f} s")
print(f"[load] 参数总量  : {n_params:,}  ({n_params/1e3:.1f} K)")
print(f"[load] 输入尺寸  : {model.config.in_channels} x "
      f"{model.config.sample_size[0]} x {model.config.sample_size[1]} (C x H x W)")
print(f"[load] 码本大小  : num_vq_embeddings={model.config.num_vq_embeddings}, "
      f"vq_embed_dim={model.config.vq_embed_dim}")

# ======================================================================
# 4. 构造合成输入(结构化的类温度场,而非纯噪声,更贴近训练分布)
#    形状 (B, C, H, W) = (1, 1, 80, 80),数值标准化到 ~N(0,1)
# ======================================================================
hr("[STEP 4] 构造测试输入并前向推理")
C = int(model.config.in_channels)
H, W = int(model.config.sample_size[0]), int(model.config.sample_size[1])

yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
field = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
         + 0.6 * np.exp(-((xx - 0.3) ** 2 + (yy - 0.4) ** 2) / 0.02)
         - 0.5 * np.exp(-((xx - 0.7) ** 2 + (yy - 0.6) ** 2) / 0.03))
field = (field - field.mean()) / (field.std() + 1e-6)
x = np.repeat(field[None, None, :, :], C, axis=1).astype(np.float32)
x = torch.from_numpy(x).to(DEVICE)
print(f"[input] 合成输入张量 shape = {tuple(x.shape)}, dtype = {x.dtype}, "
      f"range = [{x.min().item():.3f}, {x.max().item():.3f}]")

with torch.no_grad():
    enc = model.encode(x)
    latents = enc.latents if hasattr(enc, "latents") else enc[0]
print(f"[encode] 潜在表示 latents shape = {tuple(latents.shape)}  "
      f"(空间 {latents.shape[-2]}x{latents.shape[-1]} 个码本 token)")

with torch.no_grad():
    out = model(x)
    recon = out.sample if hasattr(out, "sample") else out[0]

shape_ok = tuple(recon.shape) == tuple(x.shape)
finite_ok = bool(torch.isfinite(recon).all().item())
print(f"[forward] 重构输出 shape = {tuple(recon.shape)}  "
      f"(与输入一致: {'是' if shape_ok else '否!!'})")
print(f"[forward] 输出数值有限(无 NaN/Inf): {'是' if finite_ok else '否!!'}")

# ======================================================================
# 5. 重构误差指标(输入 vs 重构)
# ======================================================================
hr("[STEP 5] 重构误差指标")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

a = x.detach().float().cpu().numpy().ravel()
b = recon.detach().float().cpu().numpy().ravel()
mae = float(mean_absolute_error(a, b))
rmse = float(np.sqrt(mean_squared_error(a, b)))
bias = float(np.mean(b - a))
r2 = float(r2_score(a, b))
corr = float(np.corrcoef(a, b)[0, 1])
max_abs = float(np.max(np.abs(b - a)))
std_err = float(np.std(b - a))

metrics = {
    "MAE": mae, "RMSE": rmse, "Bias": bias, "R2": r2,
    "Correlation": corr, "MaxAbsError": max_abs, "StdError": std_err,
}
for k, v in metrics.items():
    print(f"  {k:<14}= {v:.6f}")

# ======================================================================
# 6. 推理耗时(预热 + 多次平均)
# ======================================================================
hr("[STEP 6] 推理耗时")
with torch.no_grad():
    for _ in range(args.warmup):
        _ = model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.runs):
        _ = model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    total = time.time() - t0
per_ms = total / args.runs * 1000.0
fps = args.runs / total
print(f"  重复次数        = {args.runs} (warmup={args.warmup})")
print(f"  单次推理耗时    = {per_ms:.3f} ms/次")
print(f"  吞吐            = {fps:.2f} 次/秒")

# ======================================================================
# 7.(可选)保存可视化与数组作为证据
# ======================================================================
np.savez(os.path.join(args.model_dir, "io_arrays.npz"),
         input=x.detach().float().cpu().numpy(),
         recon=recon.detach().float().cpu().numpy())
if args.save_fig:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        im0 = ax[0].imshow(a.reshape(H, W)); ax[0].set_title("Input (synthetic tas)")
        im1 = ax[1].imshow(b.reshape(H, W)); ax[1].set_title("VQ-VAE reconstruction")
        im2 = ax[2].imshow((b - a).reshape(H, W)); ax[2].set_title("Error (recon - input)")
        for im, a_ in zip((im0, im1, im2), ax):
            fig.colorbar(im, ax=a_, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(args.model_dir, "reconstruction.png"), dpi=120)
        print("[fig] 已保存 reconstruction.png")
    except Exception as e:                                    # noqa: BLE001
        print(f"[fig] 跳过可视化({e})")

# ======================================================================
# 8. 汇总「报告块」——直接誊抄进测试报告
# ======================================================================
adaptation_pass = shape_ok and finite_ok
hr("适配测试结果汇总 (REPORT BLOCK — 誊抄用)")
print("模型            : predictia/cerra_tas_vqvae  (VQ-VAE, diffusers VQModel 结构)")
print(f"加载后端        : {BACKEND}")
print(f"设备            : {DEVICE}")
print(f"FlagGems        : {'启用' if FLAGGEMS_ON else '未启用/回退torch'}")
print(f"torch / diffusers: {torch.__version__} / {DIFFUSERS_VER}")
print(f"参数总量        : {n_params:,}")
print(f"输入 shape      : {tuple(x.shape)}")
print(f"潜在 latents shape: {tuple(latents.shape)}")
print(f"输出 shape      : {tuple(recon.shape)}  (与输入一致: {'是' if shape_ok else '否'})")
print(f"输出无NaN/Inf   : {'是' if finite_ok else '否'}")
print(f"模型加载耗时    : {load_sec:.3f} s")
print(f"单次推理耗时    : {per_ms:.3f} ms/次   吞吐 {fps:.2f} 次/秒")
print(f"重构 MAE        : {mae:.6f}")
print(f"重构 RMSE       : {rmse:.6f}")
print(f"重构 Bias       : {bias:.6f}")
print(f"重构 R2         : {r2:.6f}")
print(f"重构 Correlation: {corr:.6f}")
print(f"重构 MaxAbsError: {max_abs:.6f}")
print(f"重构 StdError   : {std_err:.6f}")
print(f"适配是否通过    : {'成功 PASS' if adaptation_pass else '失败 FAIL'}")

with open(os.path.join(args.model_dir, "result.json"), "w", encoding="utf-8") as f:
    json.dump({
        "backend": BACKEND,
        "device": DEVICE, "flaggems": FLAGGEMS_ON,
        "torch": torch.__version__, "diffusers": DIFFUSERS_VER,
        "n_params": int(n_params),
        "input_shape": list(x.shape), "latent_shape": list(latents.shape),
        "output_shape": list(recon.shape),
        "shape_ok": shape_ok, "finite_ok": finite_ok,
        "load_sec": load_sec, "infer_ms": per_ms, "fps": fps,
        "metrics": metrics, "adaptation_pass": adaptation_pass,
    }, f, ensure_ascii=False, indent=2)

print(f"\n[done] 详细日志已写入 {LOG_PATH}")
print("[done] 机器可读结果 result.json / 输入输出数组 io_arrays.npz 已保存")
sys.exit(0 if adaptation_pass else 2)
