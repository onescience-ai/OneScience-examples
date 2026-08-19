"""Validate Aurora inference arrays and render meteorological result maps.

The renderer is intentionally independent from the model and OneScience imports.  It can validate
and plot a completed inference directory on a login node with NumPy and Matplotlib only.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.resolve().parents[1] / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=None, help="Completed inference output directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for figures and validation summary")
    parser.add_argument("--variable", default=None, help="One of the 69 ERA5 channel names")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--lead-step", type=int, default=1, help="1-based forecast step")
    parser.add_argument("--format", choices=("png", "svg", "pdf"), default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def load_output(input_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing inference metadata: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = ("prediction.npy", "truth.npy", "lat.npy", "lon.npy")
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Inference output is missing {missing}")
    prediction = np.load(input_dir / "prediction.npy", mmap_mode="r")
    truth = np.load(input_dir / "truth.npy", mmap_mode="r")
    lat = np.asarray(np.load(input_dir / "lat.npy"), dtype=np.float32)
    lon = np.asarray(np.load(input_dir / "lon.npy"), dtype=np.float32)
    return prediction, truth, metadata, lat, lon


def validate_output(
    prediction: np.ndarray,
    truth: np.ndarray,
    metadata: dict[str, Any],
    lat: np.ndarray,
    lon: np.ndarray,
) -> dict[str, Any]:
    if prediction.ndim != 5 or truth.ndim != 5:
        raise ValueError(f"Prediction/truth must be [samples, steps, channels, lat, lon], got {prediction.shape}/{truth.shape}")
    if prediction.shape != truth.shape:
        raise ValueError(f"Prediction/truth shape mismatch: {prediction.shape} != {truth.shape}")
    if prediction.dtype.kind != "f" or truth.dtype.kind != "f":
        raise ValueError("Prediction and truth must use floating-point NumPy arrays")
    samples, steps, channels, height, width = prediction.shape
    channel_order = metadata.get("channel_order")
    if not isinstance(channel_order, list) or len(channel_order) != channels or len(set(channel_order)) != channels:
        raise ValueError("metadata.channel_order does not match the output channel dimension")
    units = metadata.get("units", {})
    if not isinstance(units, dict) or any(name not in units for name in channel_order):
        raise ValueError("metadata.units must contain every output channel")
    if lat.shape != (height,) or lon.shape != (width,):
        raise ValueError(f"Coordinate shape mismatch: lat={lat.shape}, lon={lon.shape}, field={(height, width)}")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("Coordinates contain NaN or infinity")
    if not np.all(np.diff(lat) < 0) or not np.all(np.diff(lon) > 0):
        raise ValueError("Expected decreasing latitude and increasing longitude")
    for sample in range(samples):
        for step in range(steps):
            for channel in range(channels):
                if not np.isfinite(prediction[sample, step, channel]).all():
                    raise ValueError("Prediction contains NaN or infinity")
                if not np.isfinite(truth[sample, step, channel]).all():
                    raise ValueError("Truth contains NaN or infinity")
    init_times = metadata.get("init_times_utc", [])
    valid_times = metadata.get("valid_times_utc", [])
    if len(init_times) != samples or len(valid_times) != samples:
        raise ValueError("Time metadata length does not match the sample dimension")
    if any(len(row) != steps for row in valid_times):
        raise ValueError("valid_times_utc rows do not match the forecast-step dimension")
    for value in init_times:
        datetime.strptime(str(value), "%Y%m%d%H")
    for row in valid_times:
        for value in row:
            datetime.strptime(str(value), "%Y%m%d%H")
    lead_times = metadata.get("lead_times_hours", [])
    if len(lead_times) != steps or any(float(value) <= 0 for value in lead_times):
        raise ValueError("lead_times_hours does not match the forecast-step dimension")
    return {
        "samples": samples,
        "forecast_steps": steps,
        "channels": channels,
        "height": height,
        "width": width,
        "finite": True,
        "latitude_order": metadata.get("latitude_order"),
        "longitude_convention": metadata.get("longitude_convention"),
    }


def compute_metrics(prediction: np.ndarray, truth: np.ndarray, channels: Sequence[str]) -> dict[str, Any]:
    samples, steps, channel_count, height, width = prediction.shape
    squared = np.zeros((steps, channel_count), dtype=np.float64)
    absolute = np.zeros((steps, channel_count), dtype=np.float64)
    bias = np.zeros((steps, channel_count), dtype=np.float64)
    for sample in range(samples):
        for step in range(steps):
            for channel in range(channel_count):
                error = (
                    np.asarray(prediction[sample, step, channel], dtype=np.float64)
                    - np.asarray(truth[sample, step, channel], dtype=np.float64)
                )
                squared[step, channel] += np.sum(error * error)
                absolute[step, channel] += np.sum(np.abs(error))
                bias[step, channel] += np.sum(error)
    denominator = float(samples * height * width)
    lead_hours = None
    return {
        "rmse": np.sqrt(squared / denominator).tolist(),
        "mae": (absolute / denominator).tolist(),
        "bias": (bias / denominator).tolist(),
        "channels": list(channels),
        "lead_count": steps,
        "lead_hours": lead_hours,
    }


def display_field(field: np.ndarray, variable: str) -> tuple[np.ndarray, str]:
    if variable == "mean_sea_level_pressure":
        return field / 100.0, "hPa"
    if variable.startswith("specific_humidity_"):
        return field * 1000.0, "g/kg"
    return field, ""


def variable_label(variable: str) -> str:
    if variable == "2m_temperature":
        return "2 m temperature"
    if variable == "mean_sea_level_pressure":
        return "Mean sea-level pressure"
    match = re.match(r"(.+)_(\d+)$", variable)
    if match:
        names = {
            "geopotential": "Geopotential",
            "u_component_of_wind": "U wind",
            "v_component_of_wind": "V wind",
            "temperature": "Temperature",
            "specific_humidity": "Specific humidity",
        }
        return f"{names.get(match.group(1), match.group(1))}, {match.group(2)} hPa"
    return variable.replace("_", " ")


def render_figure(
    prediction: np.ndarray,
    truth: np.ndarray,
    metadata: dict[str, Any],
    lat: np.ndarray,
    lon: np.ndarray,
    variable: str,
    sample_index: int,
    lead_step: int,
    output_path: Path,
    dpi: int,
    colormap: str,
    difference_colormap: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    channels = metadata["channel_order"]
    channel_index = channels.index(variable)
    lead_index = lead_step - 1
    truth_field, display_unit = display_field(np.asarray(truth[sample_index, lead_index, channel_index]), variable)
    pred_field, _ = display_field(np.asarray(prediction[sample_index, lead_index, channel_index]), variable)
    error_field = pred_field - truth_field
    vmin = float(min(truth_field.min(), pred_field.min()))
    vmax = float(max(truth_field.max(), pred_field.max()))
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    error_limit = float(np.max(np.abs(error_field)))
    if not np.isfinite(error_limit) or error_limit == 0:
        error_limit = 1.0e-6
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    absolute_map = None
    for axis, field, title in zip(axes[:2], (truth_field, pred_field), ("Truth", "Prediction")):
        absolute_map = axis.pcolormesh(
            lon, lat, field, shading="auto", cmap=colormap, vmin=vmin, vmax=vmax
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude (deg E)")
        axis.set_ylabel("Latitude (deg N)")
        axis.set_ylim(float(lat[-1]), float(lat[0]))
        axis.set_xlim(float(lon[0]), float(lon[-1]))
        axis.grid(alpha=0.25, linewidth=0.4)
    error_map = axes[2].pcolormesh(
        lon, lat, error_field, shading="auto", cmap=difference_colormap,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-error_limit, vmax=error_limit),
    )
    axes[2].set_title("Prediction - Truth")
    axes[2].set_xlabel("Longitude (deg E)")
    axes[2].set_ylabel("Latitude (deg N)")
    axes[2].set_ylim(float(lat[-1]), float(lat[0]))
    axes[2].set_xlim(float(lon[0]), float(lon[-1]))
    axes[2].grid(alpha=0.25, linewidth=0.4)
    if absolute_map is not None:
        fig.colorbar(absolute_map, ax=axes[:2], shrink=0.86, label=display_unit or metadata["units"][variable])
    fig.colorbar(error_map, ax=axes[2], shrink=0.86, label=f"Error ({display_unit or metadata['units'][variable]})")
    init_time = datetime.strptime(str(metadata["init_times_utc"][sample_index]), "%Y%m%d%H")
    valid_time = datetime.strptime(str(metadata["valid_times_utc"][sample_index][lead_index]), "%Y%m%d%H")
    lead_hours = metadata["lead_times_hours"][lead_index]
    unit_suffix = f" ({display_unit})" if display_unit else f" ({metadata['units'][variable]})"
    fig.suptitle(
        f"{variable_label(variable)}{unit_suffix} | init {init_time:%Y-%m-%d %H UTC} | "
        f"valid {valid_time:%Y-%m-%d %H UTC} | F{int(lead_hours):03d}",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, format=output_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def run_result(args: argparse.Namespace) -> Path:
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    infer_cfg = cfg.get("inference", {})
    viz_cfg = cfg.get("visualization", {})
    input_dir = resolve_path(args.input_dir or infer_cfg.get("output_dir", "outputs/inference/aurora"), config_path)
    output_dir = resolve_path(args.output_dir or viz_cfg.get("output_dir", "outputs/figures"), config_path)
    prediction, truth, metadata, lat, lon = load_output(input_dir)
    checks = validate_output(prediction, truth, metadata, lat, lon)
    channels = metadata["channel_order"]
    metrics = compute_metrics(prediction, truth, channels)
    metrics["lead_hours"] = metadata["lead_times_hours"]
    summary = {
        "schema_version": "aurora-result-summary-v1",
        "status": "validated",
        "input_dir": str(input_dir),
        "checks": checks,
        "metrics": metrics,
        "baseline": "shape/schema/range validation only; no scientific baseline is claimed",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return summary_path
    sample_index = int(args.sample_index)
    lead_step = int(args.lead_step)
    if not 0 <= sample_index < prediction.shape[0]:
        raise IndexError(f"sample-index must be in [0, {prediction.shape[0]})")
    if not 1 <= lead_step <= prediction.shape[1]:
        raise IndexError(f"lead-step must be in [1, {prediction.shape[1]}]")
    variable = args.variable or str(viz_cfg.get("default_variable", "2m_temperature"))
    if variable not in channels:
        raise ValueError(f"Unknown variable {variable!r}; choose one of the 69 configured ERA5 channels")
    file_format = args.format or str(viz_cfg.get("format", "png"))
    dpi = int(args.dpi or viz_cfg.get("dpi", 200))
    colormap = str(viz_cfg.get("colormap", "coolwarm"))
    difference_colormap = str(viz_cfg.get("difference_colormap", "RdBu_r"))
    safe_variable = re.sub(r"[^A-Za-z0-9_.-]+", "_", variable)
    figure_path = output_dir / f"sample{sample_index:04d}_F{int(metadata['lead_times_hours'][lead_step - 1]):03d}_{safe_variable}.{file_format}"
    render_figure(
        prediction,
        truth,
        metadata,
        lat,
        lon,
        variable,
        sample_index,
        lead_step,
        figure_path,
        dpi,
        colormap,
        difference_colormap,
    )
    print(json.dumps({"summary": str(summary_path), "figure": str(figure_path)}, indent=2))
    return figure_path


def main(argv: Sequence[str] | None = None) -> int:
    run_result(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
