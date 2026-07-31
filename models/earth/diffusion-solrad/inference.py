"""Platform-adapted inference for thingnario/Diffusion_SolRad."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR
RESULT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = RESULT_DIR / "predictions"

sys.path.insert(0, str(REPO_DIR))

from model_architect.UNet_DDPM import DDPM, UNet_with_time  # noqa: E402


@dataclass
class Config:
    input_frame: int = 12
    output_frame: int = 6
    cond_nc: int = 5
    time_emb_dim: int = 128
    base_chs: int = 32
    chs_mult: tuple[int, ...] = (1, 2, 4, 8, 8)
    use_attn_list: tuple[int, ...] = (0, 0, 1, 1, 1)
    n_res_blocks: int = 2
    n_steps: int = 1000
    dropout: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred-hr",
        choices=("1hr", "6hr"),
        default="1hr",
    )
    parser.add_argument(
        "--pred-mode",
        choices=("DDPM", "DDIM"),
        default="DDIM",
    )
    parser.add_argument(
        "--basetime",
        default="202504131100",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=100,
        help="Number of DDIM sampling steps.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求使用DCU，但torch.cuda.is_available()为False")
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach()

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "standard_deviation": float(tensor.std().item()),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
    }


def load_inputs(
    basetime: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample_path = (
        REPO_DIR / "sample_data" / f"sample_{basetime}.npz"
    )

    if not sample_path.exists():
        raise FileNotFoundError(f"找不到样例数据：{sample_path}")

    inputs: dict[str, torch.Tensor] = {}

    with np.load(sample_path) as data:
        for key in data.files:
            tensor = torch.from_numpy(data[key])

            if tensor.is_floating_point():
                tensor = tensor.float()

            inputs[key] = tensor.to(device)

    required = {"Himawari", "topo", "WRF", "clearsky"}
    missing = required.difference(inputs)

    if missing:
        raise KeyError(f"样例数据缺少字段：{sorted(missing)}")

    return inputs


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.ddim_steps <= 0:
        raise ValueError("--ddim-steps必须大于0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print("=" * 80)
    print("Diffusion_SolRad 平台推理")
    print("=" * 80)
    print("Torch版本：", torch.__version__)
    print("HIP版本：", torch.version.hip)
    print("推理设备：", device)

    if device.type == "cuda":
        print("设备名称：", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    inputs = load_inputs(args.basetime, device)

    print("\n输入数据：")
    input_info = {}

    for key, tensor in inputs.items():
        info = tensor_statistics(tensor)
        input_info[key] = info
        print(f"  {key}: {info}")

    config = Config()

    if args.pred_hr == "6hr":
        config.input_frame = 72
        config.output_frame = 36

    previous_himawari = inputs["Himawari"].squeeze(2)
    topography = inputs["topo"]

    model_input = torch.cat(
        [previous_himawari, topography],
        dim=1,
    )

    wrf = F.interpolate(
        inputs["WRF"].squeeze(2),
        scale_factor=4,
        mode="bilinear",
    )

    clearsky = inputs["clearsky"]

    if args.pred_hr == "1hr":
        wrf = wrf[:, :6]
        clearsky = clearsky[:, :6]

    print("\n模型输入形状：", tuple(model_input.shape))
    print("WRF条件形状：", tuple(wrf.shape))
    print("晴空辐射形状：", tuple(clearsky.shape))

    backbone = UNet_with_time(config)
    model = DDPM(
        backbone,
        output_shape=(config.output_frame, 512, 512),
    )

    checkpoint_path = (
        REPO_DIR
        / "model_weights"
        / ("ft06_01hr" if args.pred_hr == "1hr" else "ft36_06hr")
        / "weights.ckpt"
    )

    print("\n检查点：", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    load_result = model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "模型参数不一致："
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    model.eval()
    model.to(device)

    print("模型参数量：", f"{parameter_count:,}")
    print("预测时距：", args.pred_hr)
    print("采样方式：", args.pred_mode)

    if args.pred_mode == "DDIM":
        print("DDIM步数：", args.ddim_steps)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        if args.pred_mode == "DDPM":
            predicted_clearsky_index = model.sample_ddpm(
                model_input,
                input_cond=wrf,
                verbose="text",
            )
        else:
            predicted_clearsky_index = model.sample_ddim(
                model_input,
                input_cond=wrf,
                ddim_steps=args.ddim_steps,
                verbose="text",
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time

    predicted_clearsky_index = (
        predicted_clearsky_index + 1.0
    ) / 2.0

    predicted_clearsky_index = predicted_clearsky_index.clamp(
        0.0,
        1.0,
    )

    predicted_solar_radiation = (
        predicted_clearsky_index * clearsky
    )

    output_statistics = tensor_statistics(
        predicted_solar_radiation
    )

    output_name = (
        f"pred_{args.basetime}_{args.pred_hr}_"
        f"{args.pred_mode.lower()}"
    )

    if args.pred_mode == "DDIM":
        output_name += f"_{args.ddim_steps}steps"

    output_path = OUTPUT_DIR / f"{output_name}.npy"
    metrics_path = OUTPUT_DIR / f"{output_name}.json"

    np.save(
        output_path,
        predicted_solar_radiation.cpu().numpy(),
    )

    metrics: dict[str, Any] = {
        "repository": "thingnario/Diffusion_SolRad",
        "repository_revision": (
            (PROJECT_DIR / "results" / "huggingface_revision.txt")
            .read_text(encoding="utf-8")
            .strip()
            if (PROJECT_DIR / "results" / "huggingface_revision.txt").exists()
            else "unknown"
        ),
        "basetime": args.basetime,
        "random_seed": args.seed,
        "prediction_horizon": args.pred_hr,
        "sampling_mode": args.pred_mode,
        "ddim_steps": (
            args.ddim_steps
            if args.pred_mode == "DDIM"
            else None
        ),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "CPU"
        ),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "model_parameter_count": parameter_count,
        "model_config": asdict(config),
        "input_statistics": input_info,
        "output_statistics": output_statistics,
        "inference_seconds": elapsed_seconds,
        "output_file": str(output_path),
    }

    if device.type == "cuda":
        metrics["peak_allocated_memory_gib"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
        metrics["peak_reserved_memory_gib"] = (
            torch.cuda.max_memory_reserved() / 1024**3
        )

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("推理完成")
    print("=" * 80)
    print("推理耗时：", f"{elapsed_seconds:.3f} 秒")
    print("输出统计：", output_statistics)

    if device.type == "cuda":
        print(
            "峰值已分配显存：",
            f"{metrics['peak_allocated_memory_gib']:.3f} GiB",
        )
        print(
            "峰值保留显存：",
            f"{metrics['peak_reserved_memory_gib']:.3f} GiB",
        )

    print("预测文件：", output_path)
    print("指标文件：", metrics_path)

    if output_statistics["nan_count"] != 0:
        raise RuntimeError("输出中存在NaN")

    if output_statistics["inf_count"] != 0:
        raise RuntimeError("输出中存在Inf")

    print("结果有效性检查通过")


if __name__ == "__main__":
    main()
