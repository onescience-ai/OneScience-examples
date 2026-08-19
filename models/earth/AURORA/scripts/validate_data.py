"""Validate Aurora synthetic ERA5 files through OneScience ERA5Dataset."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.resolve().parents[1] / path


def decode_variables(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def validate_h5(
    path: Path,
    expected_channels: list[str],
    expected_shape: tuple[int, int],
    time_step_hours: int,
) -> dict[str, Any]:
    """Validate one HDF5 file without loading the full logical dataset."""
    with h5py.File(path, "r") as handle:
        required = {"fields", "global_means", "global_stds"}
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"{path}: missing HDF5 keys {missing}")
        fields = handle["fields"]
        variables = decode_variables(fields.attrs["variables"])
        if variables != expected_channels:
            raise ValueError(f"{path}: channel order does not match conf/config.yaml")
        if tuple(fields.shape[1:]) != (len(expected_channels), *expected_shape):
            raise ValueError(f"{path}: unexpected fields shape {fields.shape}")
        if int(fields.attrs["time_step"]) != time_step_hours:
            raise ValueError(f"{path}: unexpected time_step {fields.attrs['time_step']}")
        means = handle["global_means"][:]
        stds = handle["global_stds"][:]
        expected_stats_shape = (1, len(expected_channels), 1, 1)
        if means.shape != expected_stats_shape or stds.shape != expected_stats_shape:
            raise ValueError(f"{path}: statistics are not aligned with channels")
        minimum = float("inf")
        maximum = float("-inf")
        for step in range(fields.shape[0]):
            frame = fields[step]
            if not np.isfinite(frame).all():
                raise ValueError(f"{path}: NaN or infinity detected at time index {step}")
            minimum = min(minimum, float(frame.min()))
            maximum = max(maximum, float(frame.max()))
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError(f"{path}: NaN or infinity detected")
        if np.any(stds <= 0):
            raise ValueError(f"{path}: non-positive standard deviation detected")
        return {
            "path": str(path),
            "shape": list(fields.shape),
            "dtype": str(fields.dtype),
            "size_bytes": path.stat().st_size,
            "minimum": minimum,
            "maximum": maximum,
            "minimum_std": float(stds.min()),
        }


def validate_static(path: Path, height: int, width: int) -> dict[str, Any]:
    with np.load(path) as static:
        required = {"lsm", "z", "slt", "lat", "lon"}
        missing = sorted(required - set(static.files))
        if missing:
            raise ValueError(f"{path}: missing static arrays {missing}")
        for name in ("lsm", "z", "slt"):
            if static[name].shape != (height, width):
                raise ValueError(f"{path}: {name} has shape {static[name].shape}")
            if not np.isfinite(static[name]).all():
                raise ValueError(f"{path}: {name} contains NaN or infinity")
        if static["lat"].shape != (height,) or static["lon"].shape != (width,):
            raise ValueError(f"{path}: latitude/longitude shape mismatch")
        if not np.all(np.diff(static["lat"]) < 0) or not np.all(np.diff(static["lon"]) > 0):
            raise ValueError(f"{path}: coordinate order is incompatible with Aurora")
        return {name: list(static[name].shape) for name in required}


def validate_onescience_dataset(
    dataset_dir: Path,
    years: list[int],
    channels: list[str],
    input_steps: int,
    output_steps: int,
    normalize: bool,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Exercise the required OneScience loader and validate one sample."""
    import torch

    from onescience.datapipes.climate import ERA5Dataset

    dataset = ERA5Dataset(
        dataset_dir=str(dataset_dir),
        used_years=years,
        used_variables=channels,
        input_steps=input_steps,
        output_steps=output_steps,
        normalize=normalize,
    )
    invar, outvar, cos_zenith, step_idx, time_index = dataset[0]
    expected_input = (input_steps, len(channels), height, width)
    expected_output = (len(channels), height, width) if output_steps == 1 else (
        output_steps,
        len(channels),
        height,
        width,
    )
    if tuple(invar.shape) != expected_input:
        raise ValueError(f"OneScience input shape {tuple(invar.shape)} != {expected_input}")
    if tuple(outvar.shape) != expected_output:
        raise ValueError(f"OneScience output shape {tuple(outvar.shape)} != {expected_output}")
    if not torch.isfinite(invar).all().item() or not torch.isfinite(outvar).all().item():
        raise ValueError("OneScience sample contains NaN or infinity")
    return {
        "length": len(dataset),
        "input_shape": list(invar.shape),
        "output_shape": list(outvar.shape),
        "cos_zenith_shape": list(cos_zenith.shape),
        "first_step_index": int(step_idx),
        "first_time_index": list(time_index),
    }


def write_metadata(
    dataset_dir: Path,
    config_path: Path,
    file_stats: list[dict[str, Any]],
    split_stats: dict[str, Any],
    static_stats: dict[str, Any],
) -> None:
    metadata_dir = dataset_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    card = {
        "name": "aurora-synthetic-era5",
        "version": "1.0.0",
        "created_at": now,
        "description": "Structured synthetic ERA5 fields for Aurora workflow validation only",
        "domain": "earth",
        "format": "OneScience ERA5Dataset HDF5",
        "files": [entry["path"] for entry in file_stats],
        "static": static_stats,
        "usage": {"loader": "onescience.datapipes.climate.ERA5Dataset"},
    }
    statistics = {"files": file_stats, "splits": split_stats}
    splits = {name: {"years": value["years"], "samples": value["length"]} for name, value in split_stats.items()}
    lineage = {
        "input_sources": [],
        "processing_steps": [
            {
                "operation": "synthetic_generation",
                "script": "scripts/fake_data.py",
                "config": str(config_path),
            }
        ],
        "environment": {"python": platform.python_version()},
        "reproducibility": {
            "generate": f"python scripts/fake_data.py --config {config_path}",
            "validate": f"python scripts/validate_data.py --config {config_path}",
        },
    }
    for name, payload in (
        ("dataset_card.json", card),
        ("statistics.json", statistics),
        ("splits.json", splits),
        ("lineage.json", lineage),
    ):
        with (metadata_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    grid_cfg = data_cfg["grid"]
    dataset_dir = args.dataset_dir or resolve_project_path(data_cfg["virtual_dir"], config_path)
    dataset_dir = dataset_dir.resolve()
    height = int(grid_cfg["virtual_height"])
    width = int(grid_cfg["virtual_width"])
    channels = list(data_cfg["channel_order"])
    split_years = {
        "train": list(data_cfg["train_years"]),
        "val": list(data_cfg["val_years"]),
        "test": list(data_cfg["test_years"]),
    }

    file_stats = []
    for year in sorted({year for years in split_years.values() for year in years}):
        path = dataset_dir / "data" / f"{year}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated year: {path}")
        file_stats.append(
            validate_h5(
                path,
                channels,
                (height, width),
                int(data_cfg["time_step_hours"]),
            )
        )

    static_path = dataset_dir / "static" / "static_vars.npz"
    static_stats = validate_static(static_path, height, width)
    split_stats = {}
    for name, years in split_years.items():
        result = validate_onescience_dataset(
            dataset_dir,
            years,
            channels,
            int(data_cfg["input_steps"]),
            int(data_cfg["output_steps"]),
            bool(data_cfg["normalize_in_onescience"]),
            height,
            width,
        )
        result["years"] = years
        split_stats[name] = result
        print(f"validated {name}: {result}")

    write_metadata(dataset_dir, config_path, file_stats, split_stats, static_stats)
    print(f"dataset validation passed: {dataset_dir}")


if __name__ == "__main__":
    main()
