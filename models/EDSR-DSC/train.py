#!/usr/bin/env python3
"""Validate the local EDSR-DSC checkpoint.

Despite the historical filename ``train.py``, this script does not retrain the
model.  It loads the pretrained checkpoint, runs inference, evaluates all
available diagnostics, and writes reproducible result artifacts.

Default usage (from any working directory)::

    python train.py

Useful examples::

    python train.py --device cuda
    python train.py --data /path/to/test.nc --output-dir ./results
    python train.py --data none --height 40 --width 40

Required packages: torch, numpy
For local NetCDF data: xarray, netCDF4 (or h5netcdf)
Optional plotting: matplotlib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


SCALE = 4
N_RESBLOCKS = 16  # Inferred from the checkpoint keys; config.json incorrectly says 32.
N_FEATS = 64
N_COLORS = 2
EXPECTED_WEIGHT_SHA256 = (
    "60a0798fdd2b001ce82b3065b25e08d8179b346e96cef287c2129107cdb28d51"
)


class ResBlock(nn.Module):
    """Residual block matching super_image's EDSR checkpoint key layout."""

    def __init__(self, n_feats: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x).mul(self.res_scale)


class Upsampler(nn.Sequential):
    """Pixel-shuffle upsampler matching the EDSR x2/x3/x4 implementation."""

    def __init__(self, scale: int, n_feats: int) -> None:
        modules: list[nn.Module] = []
        if scale > 0 and (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                modules.extend(
                    [
                        nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, padding=1),
                        nn.PixelShuffle(2),
                    ]
                )
        elif scale == 3:
            modules.extend(
                [
                    nn.Conv2d(n_feats, 9 * n_feats, kernel_size=3, padding=1),
                    nn.PixelShuffle(3),
                ]
            )
        else:
            raise ValueError(f"Unsupported EDSR scale: {scale}")
        super().__init__(*modules)


class EdsrDsc(nn.Module):
    """Exact inference architecture for the local two-channel EDSR checkpoint."""

    def __init__(
        self,
        scale: int = SCALE,
        n_resblocks: int = N_RESBLOCKS,
        n_feats: int = N_FEATS,
        n_colors: int = N_COLORS,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(n_colors, n_feats, kernel_size=3, padding=1)
        )
        body: list[nn.Module] = [
            ResBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, n_colors, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(x)
        residual = self.body(x)
        residual = residual + x
        return self.tail(residual)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run EDSR-DSC inference and validation metrics."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=script_dir,
        help="Folder containing pytorch_model_4x.pt (default: script folder).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Checkpoint path (default: MODEL_DIR/pytorch_model_4x.pt).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "NetCDF path. By default the script searches MODEL_DIR recursively. "
            "Use '--data none' to force synthetic data."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result folder (default: MODEL_DIR/validation_results).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index selected from the first non-spatial NetCDF dimension.",
    )
    parser.add_argument("--height", type=int, default=40, help="Synthetic LR height.")
    parser.add_argument("--width", type=int, default=40, help="Synthetic LR width.")
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up runs.")
    parser.add_argument("--runs", type=int, default=20, help="Timed runs.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not create comparison.png."
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_state_dict(checkpoint: Any) -> OrderedDict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)!r}")

    state = OrderedDict((str(k), v) for k, v in checkpoint.items())
    if state and all(key.startswith("module.") for key in state):
        state = OrderedDict((key[7:], value) for key, value in state.items())
    if not state or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("Checkpoint is not a plain PyTorch tensor state_dict.")
    return state


def load_model(weights_path: Path, device: torch.device) -> Tuple[EdsrDsc, Dict[str, Any]]:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    checksum = sha256_file(weights_path)
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        checkpoint = torch.load(weights_path, map_location="cpu")
    state = unwrap_state_dict(checkpoint)

    model = EdsrDsc()
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_errors = [
        {
            "key": key,
            "checkpoint": list(state[key].shape),
            "model": list(model_state[key].shape),
        }
        for key in sorted(set(state) & set(model_state))
        if state[key].shape != model_state[key].shape
    ]
    if missing or unexpected or shape_errors:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"missing={missing}, unexpected={unexpected}, shape_errors={shape_errors}"
        )

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "checkpoint": str(weights_path),
        "checkpoint_sha256": checksum,
        "checksum_matches_bundled_file": checksum == EXPECTED_WEIGHT_SHA256,
        "parameter_count": parameter_count,
        "residual_blocks": N_RESBLOCKS,
        "feature_channels": N_FEATS,
        "input_channels": N_COLORS,
        "output_channels": N_COLORS,
        "scale": SCALE,
        "state_dict_key_count": len(state),
        "strict_load": True,
    }
    return model, metadata


def find_variable(dataset: Any, candidates: Sequence[str]) -> Optional[str]:
    lower_to_actual = {str(name).lower(): str(name) for name in dataset.data_vars}
    for candidate in candidates:
        match = lower_to_actual.get(candidate.lower())
        if match is not None:
            return match
    return None


def select_2d(data_array: Any, sample_index: int) -> np.ndarray:
    """Select one sample while treating the final two dimensions as spatial."""

    selected = data_array
    leading_dims = list(selected.dims[:-2])
    for position, dim in enumerate(leading_dims):
        index = sample_index if position == 0 else 0
        size = int(selected.sizes[dim])
        if index < 0 or index >= size:
            raise IndexError(
                f"sample-index {index} is outside dimension {dim!r} of size {size}"
            )
        selected = selected.isel({dim: index})
    values = np.asarray(selected.values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2-D wind field after sample selection, got {values.shape}"
        )
    return values


def discover_netcdf(model_dir: Path) -> Optional[Path]:
    preferred = model_dir / "test_data" / "test_wind_velocities.nc"
    if preferred.is_file():
        return preferred
    files = sorted(model_dir.rglob("*.nc"))
    return files[0] if files else None


def load_netcdf_sample(
    path: Path, sample_index: int
) -> Tuple[Tensor, Optional[Tensor], Dict[str, Any]]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            f"Found NetCDF data at {path}, but xarray is not installed. "
            "Run: pip install xarray netCDF4"
        ) from exc

    try:
        dataset = xr.open_dataset(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open {path}. Install a NetCDF engine with: "
            "pip install netCDF4"
        ) from exc

    try:
        u_name = find_variable(dataset, ("u100", "u100_lr", "u_lr", "u10", "u"))
        v_name = find_variable(dataset, ("v100", "v100_lr", "v_lr", "v10", "v"))
        if u_name is None or v_name is None:
            raise KeyError(
                "Could not find U/V variables. Available variables: "
                + ", ".join(map(str, dataset.data_vars))
            )

        u = select_2d(dataset[u_name], sample_index)
        v = select_2d(dataset[v_name], sample_index)
        if u.shape != v.shape:
            raise ValueError(f"U/V shapes differ: {u.shape} versus {v.shape}")
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError(
                "The selected NetCDF sample contains NaN or Inf. The model cannot "
                "accept missing values; apply the training-time mask/fill policy first."
            )

        inputs = torch.from_numpy(np.stack((u, v), axis=0)).unsqueeze(0).float()

        # Recognize common paired high-resolution variable names when present.
        u_hr_name = find_variable(
            dataset,
            ("u100_hr", "u_hr", "target_u100", "u100_target", "u100_highres"),
        )
        v_hr_name = find_variable(
            dataset,
            ("v100_hr", "v_hr", "target_v100", "v100_target", "v100_highres"),
        )
        target: Optional[Tensor] = None
        if u_hr_name is not None and v_hr_name is not None:
            u_hr = select_2d(dataset[u_hr_name], sample_index)
            v_hr = select_2d(dataset[v_hr_name], sample_index)
            if u_hr.shape != v_hr.shape:
                raise ValueError(
                    f"High-resolution U/V shapes differ: {u_hr.shape} versus {v_hr.shape}"
                )
            if not np.isfinite(u_hr).all() or not np.isfinite(v_hr).all():
                raise ValueError("High-resolution target contains NaN or Inf.")
            target = (
                torch.from_numpy(np.stack((u_hr, v_hr), axis=0))
                .unsqueeze(0)
                .float()
            )

        metadata = {
            "source_type": "local_netcdf",
            "path": str(path),
            "u_variable": u_name,
            "v_variable": v_name,
            "u_units": str(dataset[u_name].attrs.get("units", "unknown")),
            "v_units": str(dataset[v_name].attrs.get("units", "unknown")),
            "original_u_dims": list(dataset[u_name].dims),
            "original_u_shape": list(dataset[u_name].shape),
            "sample_index": sample_index,
            "has_high_resolution_target": target is not None,
            "u_hr_variable": u_hr_name,
            "v_hr_variable": v_hr_name,
        }
        return inputs, target, metadata
    finally:
        dataset.close()


def make_synthetic_sample(
    height: int, width: int
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    if height < 4 or width < 4:
        raise ValueError("Synthetic height and width must both be at least 4.")

    hr_height, hr_width = height * SCALE, width * SCALE
    y = torch.linspace(-1.0, 1.0, hr_height)
    x = torch.linspace(-1.0, 1.0, hr_width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Smooth but nontrivial deterministic vector field with multiple scales.
    radius2 = xx.square() + yy.square()
    envelope = torch.exp(-3.0 * radius2)
    u_hr = (
        5.0 * torch.sin(math.pi * xx) * torch.cos(0.8 * math.pi * yy)
        - 2.0 * yy * envelope
        + 0.6 * torch.sin(5.0 * math.pi * (xx + yy))
    )
    v_hr = (
        -4.0 * torch.cos(0.7 * math.pi * xx) * torch.sin(math.pi * yy)
        + 2.0 * xx * envelope
        + 0.5 * torch.cos(4.0 * math.pi * (xx - yy))
    )
    target = torch.stack((u_hr, v_hr), dim=0).unsqueeze(0).float()
    inputs = F.interpolate(target, size=(height, width), mode="area")
    metadata = {
        "source_type": "synthetic",
        "path": None,
        "description": "Deterministic analytic vector field; metrics are smoke-test only.",
        "has_high_resolution_target": True,
        "sample_index": 0,
    }
    return inputs, target, metadata


def choose_data(
    model_dir: Path,
    data_arg: Optional[str],
    sample_index: int,
    height: int,
    width: int,
) -> Tuple[Tensor, Optional[Tensor], Dict[str, Any]]:
    if data_arg is not None and data_arg.strip().lower() == "none":
        return make_synthetic_sample(height, width)

    if data_arg is not None:
        data_path = Path(data_arg).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"Requested data file does not exist: {data_path}")
    else:
        data_path = discover_netcdf(model_dir)

    if data_path is not None:
        return load_netcdf_sample(data_path, sample_index)

    print("No NetCDF data found; generating deterministic synthetic test data.")
    return make_synthetic_sample(height, width)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_inference(
    model: nn.Module,
    inputs: Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
) -> Tuple[Tensor, Dict[str, Any]]:
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")
    inputs_device = inputs.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(inputs_device)
        synchronize(device)

        timings_ms = []
        output: Optional[Tensor] = None
        for _ in range(runs):
            synchronize(device)
            start = time.perf_counter()
            output = model(inputs_device)
            synchronize(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    assert output is not None
    performance: Dict[str, Any] = {
        "device": str(device),
        "warmup_runs": warmup,
        "timed_runs": runs,
        "latency_ms_mean": float(np.mean(timings_ms)),
        "latency_ms_median": float(np.median(timings_ms)),
        "latency_ms_min": float(np.min(timings_ms)),
        "latency_ms_max": float(np.max(timings_ms)),
    }
    if device.type == "cuda":
        performance["gpu_name"] = torch.cuda.get_device_name(device)
        performance["peak_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
    return output.detach().cpu(), performance


def scalar_metrics(prediction: Tensor, target: Tensor) -> Dict[str, float]:
    prediction = prediction.float().cpu()
    target = target.float().cpu()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shapes differ: {tuple(prediction.shape)} versus "
            f"{tuple(target.shape)}"
        )
    error = prediction - target
    speed_prediction = torch.linalg.vector_norm(prediction, dim=1)
    speed_target = torch.linalg.vector_norm(target, dim=1)
    speed_error = speed_prediction - speed_target
    return {
        "mae_uv_mean": float(error.abs().mean()),
        "rmse_uv_mean": float(error.square().mean().sqrt()),
        "mae_u": float(error[:, 0].abs().mean()),
        "rmse_u": float(error[:, 0].square().mean().sqrt()),
        "mae_v": float(error[:, 1].abs().mean()),
        "rmse_v": float(error[:, 1].square().mean().sqrt()),
        "vector_rmse": float(
            (error[:, 0].square() + error[:, 1].square()).mean().sqrt()
        ),
        "speed_mae": float(speed_error.abs().mean()),
        "speed_rmse": float(speed_error.square().mean().sqrt()),
        "max_abs_error": float(error.abs().max()),
    }


def gradient_energy(field: Tensor) -> float:
    dx = field[..., :, 1:] - field[..., :, :-1]
    dy = field[..., 1:, :] - field[..., :-1, :]
    return float(0.5 * (dx.abs().mean() + dy.abs().mean()))


def evaluate(
    inputs: Tensor, prediction: Tensor, target: Optional[Tensor]
) -> Tuple[Tensor, Dict[str, Any]]:
    expected_shape = (
        inputs.shape[0],
        N_COLORS,
        inputs.shape[-2] * SCALE,
        inputs.shape[-1] * SCALE,
    )
    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(
            f"Wrong output shape: got {tuple(prediction.shape)}, expected {expected_shape}"
        )

    finite_fraction = float(torch.isfinite(prediction).float().mean())
    if finite_fraction != 1.0:
        raise RuntimeError(f"Model output contains NaN/Inf: finite={finite_fraction:.6f}")

    baseline = F.interpolate(
        inputs, scale_factor=SCALE, mode="bicubic", align_corners=False
    )
    downsampled_prediction = F.interpolate(
        prediction, size=inputs.shape[-2:], mode="area"
    )
    metrics: Dict[str, Any] = {
        "status": "PASS",
        "validation_level": (
            "paired_target" if target is not None else "functional_only"
        ),
        "input_shape": list(inputs.shape),
        "output_shape": list(prediction.shape),
        "output_finite_fraction": finite_fraction,
        "output_u_min": float(prediction[:, 0].min()),
        "output_u_max": float(prediction[:, 0].max()),
        "output_u_mean": float(prediction[:, 0].mean()),
        "output_u_std": float(prediction[:, 0].std()),
        "output_v_min": float(prediction[:, 1].min()),
        "output_v_max": float(prediction[:, 1].max()),
        "output_v_mean": float(prediction[:, 1].mean()),
        "output_v_std": float(prediction[:, 1].std()),
        "self_consistency_area_downsample": scalar_metrics(
            downsampled_prediction, inputs
        ),
        "model_gradient_energy": gradient_energy(prediction),
        "bicubic_gradient_energy": gradient_energy(baseline),
    }
    denominator = metrics["bicubic_gradient_energy"]
    metrics["detail_energy_ratio_vs_bicubic"] = (
        metrics["model_gradient_energy"] / denominator
        if denominator > 0
        else None
    )

    if target is not None:
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"High-resolution target has shape {tuple(target.shape)}; expected "
                f"{expected_shape}. Check time and grid alignment."
            )
        model_metrics = scalar_metrics(prediction, target)
        baseline_metrics = scalar_metrics(baseline, target)
        metrics["model_vs_target"] = model_metrics
        metrics["bicubic_vs_target"] = baseline_metrics
        baseline_rmse = baseline_metrics["rmse_uv_mean"]
        metrics["rmse_skill_vs_bicubic"] = (
            1.0 - model_metrics["rmse_uv_mean"] / baseline_rmse
            if baseline_rmse > 0
            else None
        )
    else:
        metrics["scientific_accuracy_note"] = (
            "No paired high-resolution ground truth was found. Shape, finiteness, "
            "self-consistency, speed, and memory are valid diagnostics, but MAE/RMSE "
            "against real COSMO-REA6 fields cannot be computed."
        )
    return baseline, metrics


def save_artifacts(
    output_dir: Path,
    inputs: Tensor,
    prediction: Tensor,
    baseline: Tensor,
    target: Optional[Tensor],
    report: Dict[str, Any],
    make_plot: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {
        "u_lr": inputs[0, 0].numpy(),
        "v_lr": inputs[0, 1].numpy(),
        "u_sr": prediction[0, 0].numpy(),
        "v_sr": prediction[0, 1].numpy(),
        "u_bicubic": baseline[0, 0].numpy(),
        "v_bicubic": baseline[0, 1].numpy(),
    }
    if target is not None:
        arrays["u_hr_target"] = target[0, 0].numpy()
        arrays["v_hr_target"] = target[0, 1].numpy()
    np.savez_compressed(output_dir / "prediction.npz", **arrays)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    # NetCDF output is convenient when xarray is available, but NPZ is always saved.
    try:
        import xarray as xr

        data_vars: Dict[str, Any] = {
            "u100_lr": (("y_lr", "x_lr"), arrays["u_lr"]),
            "v100_lr": (("y_lr", "x_lr"), arrays["v_lr"]),
            "u100_sr": (("y_hr", "x_hr"), arrays["u_sr"]),
            "v100_sr": (("y_hr", "x_hr"), arrays["v_sr"]),
            "u100_bicubic": (("y_hr", "x_hr"), arrays["u_bicubic"]),
            "v100_bicubic": (("y_hr", "x_hr"), arrays["v_bicubic"]),
        }
        if target is not None:
            data_vars["u100_hr_target"] = (
                ("y_hr", "x_hr"),
                arrays["u_hr_target"],
            )
            data_vars["v100_hr_target"] = (
                ("y_hr", "x_hr"),
                arrays["v_hr_target"],
            )
        xr.Dataset(
            data_vars,
            attrs={
                "model": "lschmidt/edsr-dsc",
                "scale_factor": SCALE,
                "coordinate_note": (
                    "Index coordinates only; no unverified high-resolution geographic "
                    "coordinates were invented."
                ),
            },
        ).to_netcdf(output_dir / "prediction.nc")
    except Exception as exc:
        print(f"Warning: prediction.nc was not written: {exc}")

    if make_plot:
        save_plot(output_dir / "comparison.png", inputs, prediction, baseline, target)


def save_plot(
    path: Path,
    inputs: Tensor,
    prediction: Tensor,
    baseline: Tensor,
    target: Optional[Tensor],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is unavailable; comparison.png was not written.")
        return

    def speed(tensor: Tensor) -> np.ndarray:
        return torch.linalg.vector_norm(tensor[0], dim=0).numpy()

    speed_lr = speed(inputs)
    speed_bicubic = speed(baseline)
    speed_prediction = speed(prediction)
    if target is not None:
        speed_reference = speed(target)
        fourth = np.abs(speed_prediction - speed_reference)
        fourth_title = "|EDSR speed - target speed|"
    else:
        speed_reference = speed_bicubic
        fourth = np.abs(speed_prediction - speed_bicubic)
        fourth_title = "|EDSR speed - bicubic speed|"

    color_min = float(min(speed_lr.min(), speed_reference.min(), speed_prediction.min()))
    color_max = float(max(speed_lr.max(), speed_reference.max(), speed_prediction.max()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = (
        (speed_lr, "Low-resolution wind speed", color_min, color_max),
        (speed_bicubic, "Bicubic x4 wind speed", color_min, color_max),
        (speed_prediction, "EDSR x4 wind speed", color_min, color_max),
        (fourth, fourth_title, 0.0, float(fourth.max())),
    )
    for axis, (array, title, vmin, vmax) in zip(axes.flat, panels):
        image = axis.imshow(array, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(f"{title}  {array.shape}")
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def print_summary(report: Dict[str, Any], output_dir: Path) -> None:
    metrics = report["metrics"]
    performance = report["performance"]
    print("\n========== EDSR-DSC validation ==========")
    print(f"Status:              {metrics['status']}")
    print(f"Validation level:    {metrics['validation_level']}")
    print(f"Data source:         {report['data']['source_type']}")
    print(f"Device:              {performance['device']}")
    print(f"Input shape:         {tuple(metrics['input_shape'])}")
    print(f"Output shape:        {tuple(metrics['output_shape'])}")
    print(f"Finite output:       {metrics['output_finite_fraction']:.6f}")
    print(f"Median latency:      {performance['latency_ms_median']:.3f} ms")
    if "peak_memory_mb" in performance:
        print(f"Peak GPU memory:     {performance['peak_memory_mb']:.2f} MB")
    consistency = metrics["self_consistency_area_downsample"]
    print(f"Self-consistency MAE:{consistency['mae_uv_mean']:.6f}")
    print(f"Self-consistency RMSE:{consistency['rmse_uv_mean']:.6f}")
    if "model_vs_target" in metrics:
        model_rmse = metrics["model_vs_target"]["rmse_uv_mean"]
        baseline_rmse = metrics["bicubic_vs_target"]["rmse_uv_mean"]
        print(f"Model target RMSE:   {model_rmse:.6f}")
        print(f"Bicubic target RMSE:{baseline_rmse:.6f}")
        print(f"RMSE skill:          {metrics['rmse_skill_vs_bicubic']:.6f}")
    else:
        print("Scientific accuracy: unavailable (no paired HR target)")
    print(f"Results:             {output_dir}")
    print("=========================================\n")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = args.model_dir.expanduser().resolve()
    weights_path = (
        args.weights.expanduser().resolve()
        if args.weights is not None
        else model_dir / "pytorch_model_4x.pt"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir / "validation_results"
    )
    device = resolve_device(args.device)

    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")

    model, model_metadata = load_model(weights_path, device)
    inputs, target, data_metadata = choose_data(
        model_dir=model_dir,
        data_arg=args.data,
        sample_index=args.sample_index,
        height=args.height,
        width=args.width,
    )
    prediction, performance = run_inference(
        model=model,
        inputs=inputs,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
    )
    baseline, metrics = evaluate(inputs, prediction, target)

    report: Dict[str, Any] = {
        "status": "PASS",
        "created_unix_time": time.time(),
        "seed": args.seed,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "model": model_metadata,
        "data": data_metadata,
        "performance": performance,
        "metrics": metrics,
    }
    save_artifacts(
        output_dir=output_dir,
        inputs=inputs,
        prediction=prediction,
        baseline=baseline,
        target=target,
        report=report,
        make_plot=not args.no_plot,
    )
    print_summary(report, output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
