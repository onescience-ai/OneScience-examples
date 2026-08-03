#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
openclimatefix/dgmr 单样本推理复现脚本
=============================================

功能
----
1. 从 Hugging Face 下载并加载 DGMR 预训练权重；
2. 自动生成 4 帧 256×256 单通道仿真雷达序列；
3. 使用生成器预测未来 18 帧；
4. 保存 NPY、PNG、GIF 和 JSON 报告；
5. 可选启用 FlagGems，默认关闭。

重要说明
--------
仿真输入只用于验证模型下载、加载和前向推理流程。它不是真实雷达数据，
因此模型原始输出不能直接解释为 mm/h，也不能用于复现论文业务精度。

推荐环境
--------
Python 3.10
保留平台已有的定制 PyTorch，不要让 pip 覆盖 torch/torchvision。

Notebook 运行
-------------
%cd /root/private_data/whh/dgmr
%run reproduce_dgmr.py

可选 FlagGems：
%run reproduce_dgmr.py --flag-gems

生成 3 个随机集合成员：
%run reproduce_dgmr.py --members 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

MODEL_ID = "openclimatefix/dgmr"
EXPECTED_MODEL_ID = "openclimatefix/dgmr"
HISTORY_STEPS = 4
FORECAST_STEPS = 18
HEIGHT = 256
WIDTH = 256
CHANNELS = 1
SEED = 42


class ReproductionError(RuntimeError):
    """可读性较强的复现错误。"""


def check_python() -> None:
    if sys.version_info[:2] != (3, 10):
        raise ReproductionError(
            f"当前 Python 为 {platform.python_version()}；"
            "本复现方案按 Python 3.10 环境编写。"
        )


def check_imports() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    import_error: Exception | None = None

    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")

    try:
        import torch
    except ImportError:
        torch = None
        missing.append("torch")

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        missing.append("matplotlib")

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("pillow")

    try:
        from dgmr import DGMR
    except ImportError as exc:
        DGMR = None
        missing.append("dgmr 及其推理依赖")
        import_error = exc

    if missing:
        details = "\n".join(f"  - {name}" for name in missing)
        message = (
            "当前环境缺少以下依赖：\n"
            f"{details}\n\n"
            "请先按 README 中的 Notebook 安装命令安装依赖。"
        )
        if import_error is not None:
            message += (
                "\nDGMR 导入错误："
                f"{type(import_error).__name__}: {import_error}"
            )
        raise ReproductionError(message)

    return np, torch, DGMR


def enable_flag_gems(base_dir: Path, enabled: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "log_path": None,
        "unused": [],
    }

    if not enabled:
        print("FlagGems：未启用（原生 PyTorch 模式）")
        return info

    try:
        import flag_gems
    except ImportError as exc:
        raise ReproductionError(
            "指定了 --flag-gems，但当前环境无法导入 flag_gems。"
        ) from exc

    log_path = base_dir / "gems_debug.log"
    unused = [
        "batch_norm",
        "batch_norm_backward",
    ]

    flag_gems.enable(
        unused=unused,
        record=True,
        path=str(log_path),
        once=True,
    )

    info.update(
        {
            "enabled": True,
            "log_path": str(log_path),
            "unused": unused,
        }
    )

    print("FlagGems：已启用")
    print("FlagGems 日志：", log_path)
    return info


def gaussian_blob(
    np: Any,
    xx: Any,
    yy: Any,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    amplitude: float,
) -> Any:
    exponent = -0.5 * (
        ((xx - center_x) / sigma_x) ** 2
        + ((yy - center_y) / sigma_y) ** 2
    )
    return amplitude * np.exp(exponent)


def generate_synthetic_radar(np: Any, seed: int = SEED) -> Any:
    """
    生成 4 帧平滑、移动的雷达回波场。

    返回：
        shape = (1, 4, 1, 256, 256)
        dtype = float32
        range = 0..1
    """
    rng = np.random.default_rng(seed)
    y = np.arange(HEIGHT, dtype=np.float32)
    x = np.arange(WIDTH, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    frames: list[Any] = []

    for t in range(HISTORY_STEPS):
        field = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

        field += gaussian_blob(
            np,
            xx,
            yy,
            center_x=58 + 9.0 * t,
            center_y=68 + 5.0 * t,
            sigma_x=20,
            sigma_y=15,
            amplitude=0.92,
        )

        field += gaussian_blob(
            np,
            xx,
            yy,
            center_x=174 - 4.0 * t,
            center_y=164 - 7.0 * t,
            sigma_x=31,
            sigma_y=22,
            amplitude=0.63,
        )

        field += gaussian_blob(
            np,
            xx,
            yy,
            center_x=215 - 2.0 * t,
            center_y=63 + 4.0 * t,
            sigma_x=13,
            sigma_y=12,
            amplitude=0.42,
        )

        # 斜向雨带
        distance = np.abs(
            (yy - (126 + 2.0 * t))
            - 0.24 * (xx - 128)
        )
        band = np.exp(-(distance / 9.5) ** 2)
        band *= np.exp(-((xx - (126 + 4.0 * t)) / 92.0) ** 2)
        field += 0.25 * band

        # 小幅扰动
        noise = rng.normal(0.0, 0.012, size=(HEIGHT, WIDTH))
        field += noise.astype(np.float32)

        field = np.clip(field, 0.0, 1.0)
        frames.append(field.astype(np.float32))

    array = np.stack(frames, axis=0)     # T,H,W
    array = array[:, None, :, :]         # T,C,H,W
    array = array[None, :, :, :, :]      # B,T,C,H,W
    return array.astype(np.float32)


def display_limits(np: Any, frames: Any) -> tuple[float, float]:
    finite = frames[np.isfinite(frames)]

    if finite.size == 0:
        return 0.0, 1.0

    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))

    if (
        not math.isfinite(low)
        or not math.isfinite(high)
        or high <= low
    ):
        low = float(finite.min())
        high = float(finite.max()) + 1e-6

    return low, high


def save_montage(
    np: Any,
    plt: Any,
    frames: Any,
    output_path: Path,
    title: str,
    *,
    columns: int,
) -> None:
    """把 T×H×W 序列画成拼图。"""
    total = int(frames.shape[0])
    rows = int(math.ceil(total / columns))
    low, high = display_limits(np, frames)

    figure = plt.figure(figsize=(3.0 * columns, 2.75 * rows))

    for index in range(total):
        axis = figure.add_subplot(rows, columns, index + 1)
        axis.imshow(
            frames[index],
            origin="lower",
            vmin=low,
            vmax=high,
        )
        axis.set_title(f"Frame {index + 1}")
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def normalize_for_display(np: Any, frames: Any) -> Any:
    low, high = display_limits(np, frames)
    scaled = (frames - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).round().astype(np.uint8)


def save_gif(
    np: Any,
    image_class: Any,
    frames: Any,
    output_path: Path,
) -> None:
    display = normalize_for_display(np, frames)
    images = [image_class.fromarray(frame) for frame in display]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=350,
        loop=0,
    )


def normalize_prediction_shape(
    torch: Any,
    prediction: Any,
) -> Any:
    """统一验证为 (B,T,C,H,W)。"""
    if isinstance(prediction, (tuple, list)):
        if not prediction:
            raise ReproductionError("模型返回了空列表或空元组。")
        prediction = prediction[0]

    if not torch.is_tensor(prediction):
        raise ReproductionError(
            f"模型输出类型异常：{type(prediction).__name__}"
        )

    if prediction.ndim != 5:
        raise ReproductionError(
            f"模型输出应为 5 维，实际为 {tuple(prediction.shape)}"
        )

    return prediction


def run_inference(
    torch: Any,
    generator: Any,
    input_array: Any,
    device: str,
    members: int,
) -> tuple[list[Any], list[float]]:
    predictions: list[Any] = []
    elapsed_times: list[float] = []

    input_tensor = torch.from_numpy(input_array).to(device=device)

    for member_index in range(members):
        member_seed = SEED + member_index
        torch.manual_seed(member_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(member_seed)
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            prediction = generator(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        prediction = normalize_prediction_shape(torch, prediction)
        predictions.append(prediction.detach().float().cpu().numpy())
        elapsed_times.append(elapsed)

        print(
            f"集合成员 {member_index + 1}/{members} 完成，"
            f"形状={tuple(prediction.shape)}，耗时={elapsed:.3f} 秒"
        )

    return predictions, elapsed_times


def main() -> int:
    parser = argparse.ArgumentParser(
        description="openclimatefix/dgmr 单样本推理复现"
    )
    parser.add_argument(
        "--members",
        type=int,
        default=1,
        help="随机集合成员数量，默认 1。",
    )
    parser.add_argument(
        "--flag-gems",
        action="store_true",
        help="在当前进程中启用 FlagGems。",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="强制使用 CPU；完整 256×256 模型会很慢。",
    )
    args = parser.parse_args()

    if args.members < 1:
        raise ReproductionError("--members 必须至少为 1。")

    check_python()

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "dgmr_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 把 Hugging Face 缓存固定到项目目录。
    hf_home = base_dir / "hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault(
        "HUGGINGFACE_HUB_CACHE",
        str(hf_home / "hub"),
    )

    np, torch, DGMR = check_imports()

    from PIL import Image
    import matplotlib.pyplot as plt

    print("=" * 72)
    print("DGMR 推理复现")
    print("=" * 72)
    print("Python：", platform.python_version())
    print("Python 路径：", sys.executable)
    print("PyTorch：", torch.__version__)
    print("CUDA 可用：", torch.cuda.is_available())
    print("Hugging Face 模型：", MODEL_ID)
    print("Hugging Face 缓存：", hf_home)

    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print("运行设备：", device)

    if device.startswith("cuda"):
        print("设备名称：", torch.cuda.get_device_name(0))

    flag_gems_info = enable_flag_gems(base_dir, args.flag_gems)

    input_array = generate_synthetic_radar(np)
    input_npy = output_dir / "synthetic_radar_input.npy"
    input_png = output_dir / "synthetic_radar_input.png"

    np.save(input_npy, input_array)
    save_montage(
        np,
        plt,
        input_array[0, :, 0],
        input_png,
        "Synthetic DGMR Radar Input",
        columns=4,
    )

    print("仿真输入形状：", input_array.shape)
    print(
        "仿真输入范围：",
        float(input_array.min()),
        float(input_array.max()),
    )

    if MODEL_ID != EXPECTED_MODEL_ID:
        raise ReproductionError(
            "模型地址不符合本次测试要求："
            f"{MODEL_ID} != {EXPECTED_MODEL_ID}"
        )

    print("\n正在下载/加载预训练模型……")
    load_start = time.perf_counter()

    try:
        full_model = DGMR.from_pretrained(MODEL_ID)
    except Exception as exc:
        raise ReproductionError(
            "DGMR 预训练模型下载或加载失败。\n"
            "请确认当前环境可访问 Hugging Face，且 dgmr、"
            "huggingface_hub、pytorch_lightning 等依赖已正确安装。\n"
            f"原始错误：{type(exc).__name__}: {exc}"
        ) from exc

    total_parameter_count = sum(
        int(parameter.numel())
        for parameter in full_model.parameters()
    )

    # 推理只需要生成器，避免把判别器也移动到加速设备。
    generator = full_model.generator.to(device)
    generator.eval()

    generator_parameter_count = sum(
        int(parameter.numel())
        for parameter in generator.parameters()
    )

    load_seconds = time.perf_counter() - load_start

    print(f"模型加载完成，耗时 {load_seconds:.3f} 秒")
    print("完整模型参数量：", total_parameter_count)
    print("生成器参数量：", generator_parameter_count)

    predictions, elapsed_times = run_inference(
        torch,
        generator,
        input_array,
        device,
        args.members,
    )

    prediction_stack = np.concatenate(predictions, axis=0)

    expected_shape = (
        args.members,
        FORECAST_STEPS,
        CHANNELS,
        HEIGHT,
        WIDTH,
    )

    if prediction_stack.shape != expected_shape:
        raise ReproductionError(
            "预测形状与预期不一致："
            f"{prediction_stack.shape} != {expected_shape}"
        )

    prediction_npy = output_dir / "prediction_members.npy"
    ensemble_mean_npy = output_dir / "prediction_ensemble_mean.npy"
    prediction_png = output_dir / "prediction_member_00.png"
    prediction_gif = output_dir / "prediction_member_00.gif"

    np.save(prediction_npy, prediction_stack)

    ensemble_mean = prediction_stack.mean(axis=0)
    np.save(ensemble_mean_npy, ensemble_mean)

    first_member = prediction_stack[0, :, 0]

    save_montage(
        np,
        plt,
        first_member,
        prediction_png,
        "DGMR Forecast: Ensemble Member 1",
        columns=6,
    )

    save_gif(
        np,
        Image,
        first_member,
        prediction_gif,
    )

    finite_mask = np.isfinite(prediction_stack)
    has_nan = bool(np.isnan(prediction_stack).any())
    has_inf = bool(np.isinf(prediction_stack).any())

    report = {
        "model_id": MODEL_ID,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if device.startswith("cuda")
            else None
        ),
        "flag_gems": flag_gems_info,
        "hf_home": str(hf_home),
        "model_load_seconds": round(load_seconds, 6),
        "total_parameter_count": total_parameter_count,
        "generator_parameter_count": generator_parameter_count,
        "input_shape": list(input_array.shape),
        "input_dtype": str(input_array.dtype),
        "input_min": float(input_array.min()),
        "input_max": float(input_array.max()),
        "members": args.members,
        "prediction_shape": list(prediction_stack.shape),
        "prediction_dtype": str(prediction_stack.dtype),
        "prediction_min": (
            float(prediction_stack[finite_mask].min())
            if finite_mask.any()
            else None
        ),
        "prediction_max": (
            float(prediction_stack[finite_mask].max())
            if finite_mask.any()
            else None
        ),
        "prediction_mean": (
            float(prediction_stack[finite_mask].mean())
            if finite_mask.any()
            else None
        ),
        "prediction_std": (
            float(prediction_stack[finite_mask].std())
            if finite_mask.any()
            else None
        ),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "inference_seconds_per_member": [
            round(value, 6) for value in elapsed_times
        ],
        "files": {
            "input_npy": str(input_npy),
            "input_png": str(input_png),
            "prediction_members_npy": str(prediction_npy),
            "ensemble_mean_npy": str(ensemble_mean_npy),
            "prediction_png": str(prediction_png),
            "prediction_gif": str(prediction_gif),
        },
        "interpretation_warning": (
            "仿真输入仅验证工程推理流程；模型输出为原始网络尺度，"
            "不能直接解释为真实降雨率，也不能用于复现论文业务指标。"
        ),
        "success": (
            prediction_stack.shape == expected_shape
            and not has_nan
            and not has_inf
        ),
    }

    report_path = output_dir / "reproduction_report.json"
    report["files"]["report_json"] = str(report_path)

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("复现完成")
    print("=" * 72)
    print("预测形状：", prediction_stack.shape)
    print("预测最小值：", report["prediction_min"])
    print("预测最大值：", report["prediction_max"])
    print("预测均值：", report["prediction_mean"])
    print("存在 NaN：", has_nan)
    print("存在 Inf：", has_inf)
    print("结果目录：", output_dir)
    print("报告文件：", report_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中止运行。", file=sys.stderr)
        raise SystemExit(130)
    except ReproductionError as exc:
        print(f"\n复现未完成：{exc}", file=sys.stderr)
        raise SystemExit(1)
