"""Generate a compact, structured ERA5 dataset for Aurora workflow tests."""

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
    """Load the project YAML configuration."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(value: str, config_path: Path) -> Path:
    """Resolve a configured path relative to the project containing the config."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return config_path.resolve().parents[1] / path


def make_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create Aurora-compatible decreasing latitudes and increasing longitudes."""
    lat = np.linspace(90.0, -90.0, height, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, width, endpoint=False, dtype=np.float32)
    lat_radians = np.deg2rad(lat)[:, None]
    lon_radians = np.deg2rad(lon)[None, :]
    return lat, lon, lat_radians, lon_radians


def make_static_fields(height: int, width: int) -> dict[str, np.ndarray]:
    """Construct deterministic land mask, geopotential, and soil-type fields."""
    lat, lon, lat_radians, lon_radians = make_grid(height, width)
    continent = np.sin(1.7 * lon_radians) + 0.55 * np.cos(2.4 * lat_radians)
    lsm = (continent > 0.15).astype(np.float32)
    elevation = np.maximum(
        0.0,
        1800.0 * np.cos(lat_radians) ** 2 * (0.55 + 0.45 * np.cos(2.0 * lon_radians)),
    )
    z = (elevation * 9.80665 * lsm).astype(np.float32)
    soil_pattern = (np.floor((lon[None, :] / 45.0) + (lat[:, None] + 90.0) / 30.0) % 8)
    slt = np.where(lsm > 0, soil_pattern, 0).astype(np.float32)
    return {"lsm": lsm, "z": z, "slt": slt, "lat": lat, "lon": lon}


def pressure_from_name(name: str) -> int:
    """Extract the pressure level suffix from a configured ERA5 channel name."""
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Pressure-level channel has no integer suffix: {name}") from exc


def make_dynamic_field(
    name: str,
    step: int,
    year: int,
    lat_radians: np.ndarray,
    lon_radians: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create one physically scaled synthetic field without emulating forecast skill."""
    phase = 2.0 * np.pi * step / 4.0
    planetary = np.cos(lat_radians) * np.sin(lon_radians + phase)
    synoptic = np.sin(2.0 * lat_radians + 0.5 * phase) * np.cos(2.0 * lon_radians)
    noise = rng.standard_normal(planetary.shape).astype(np.float32)
    year_offset = float(year - 2000)

    if name == "2m_temperature":
        field = 273.15 + 24.0 * np.cos(lat_radians) + 2.5 * planetary + 0.02 * year_offset
        return (field + 0.15 * noise).astype(np.float32)
    if name == "10m_u_component_of_wind":
        return (12.0 * planetary + 2.0 * synoptic + 0.2 * noise).astype(np.float32)
    if name == "10m_v_component_of_wind":
        return (8.0 * synoptic - 1.5 * planetary + 0.2 * noise).astype(np.float32)
    if name == "mean_sea_level_pressure":
        field = 101325.0 + 1300.0 * synoptic + 450.0 * planetary
        return (field + 20.0 * noise).astype(np.float32)

    level = pressure_from_name(name)
    pressure_ratio = level / 1000.0
    if name.startswith("geopotential_"):
        altitude = 44330.0 * (1.0 - pressure_ratio**0.1903)
        field = 9.80665 * altitude + 120.0 * planetary + 30.0 * synoptic
        return (field + 4.0 * noise).astype(np.float32)
    if name.startswith("u_component_of_wind_"):
        scale = 8.0 + 16.0 * (1.0 - pressure_ratio)
        return (scale * planetary + 3.0 * synoptic + 0.25 * noise).astype(np.float32)
    if name.startswith("v_component_of_wind_"):
        scale = 6.0 + 12.0 * (1.0 - pressure_ratio)
        return (scale * synoptic - 2.0 * planetary + 0.25 * noise).astype(np.float32)
    if name.startswith("temperature_"):
        reference = 288.0 * pressure_ratio**0.1903
        field = reference + 7.0 * np.cos(lat_radians) + 1.5 * planetary
        return (field + 0.1 * noise).astype(np.float32)
    if name.startswith("specific_humidity_"):
        reference = 0.012 * pressure_ratio**1.6
        field = reference * (0.75 + 0.25 * np.cos(lat_radians))
        field = field + 0.00025 * planetary + 0.00001 * noise
        return np.maximum(field, 1.0e-8).astype(np.float32)
    raise ValueError(f"Unsupported Aurora ERA5 channel: {name}")


def generate_year(
    path: Path,
    channels: list[str],
    timesteps: int,
    height: int,
    width: int,
    time_step_hours: int,
    seed: int,
    year: int,
) -> dict[str, Any]:
    """Generate one HDF5 year and return compact statistics."""
    _, _, lat_radians, lon_radians = make_grid(height, width)
    path.parent.mkdir(parents=True, exist_ok=True)
    sums = np.zeros(len(channels), dtype=np.float64)
    squared_sums = np.zeros(len(channels), dtype=np.float64)
    count_per_channel = timesteps * height * width

    chunk_height = min(height, 180)
    chunk_width = min(width, 360)
    with h5py.File(path, "w") as handle:
        fields = handle.create_dataset(
            "fields",
            shape=(timesteps, len(channels), height, width),
            dtype=np.float32,
            chunks=(1, 1, chunk_height, chunk_width),
            compression="gzip",
            compression_opts=1,
        )
        fields.attrs["variables"] = np.asarray(channels, dtype=h5py.string_dtype("utf-8"))
        fields.attrs["time_step"] = time_step_hours
        fields.attrs["synthetic"] = True
        fields.attrs["generator"] = "AURORA/scripts/fake_data.py"

        for step in range(timesteps):
            rng = np.random.default_rng(seed + year * 1009 + step)
            for channel_index, channel in enumerate(channels):
                field = make_dynamic_field(
                    channel, step, year, lat_radians, lon_radians, rng
                )
                fields[step, channel_index] = field
                sums[channel_index] += field.sum(dtype=np.float64)
                squared_sums[channel_index] += np.square(
                    field, dtype=np.float64
                ).sum(dtype=np.float64)

        means = sums / count_per_channel
        variance = np.maximum(squared_sums / count_per_channel - means**2, 1.0e-12)
        stds = np.sqrt(variance)
        handle.create_dataset("global_means", data=means[None, :, None, None].astype(np.float32))
        handle.create_dataset("global_stds", data=stds[None, :, None, None].astype(np.float32))

    return {
        "path": str(path),
        "shape": [timesteps, len(channels), height, width],
        "size_bytes": path.stat().st_size,
        "finite": True,
        "minimum_std": float(stds.min()),
    }


def write_dataset_metadata(
    output_dir: Path,
    config_path: Path,
    generated: list[dict[str, Any]],
    split_years: dict[str, list[int]],
    static_fields: dict[str, np.ndarray],
    input_steps: int,
    output_steps: int,
) -> None:
    """Write metadata describing the generated synthetic dataset."""
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    static_shapes = {name: list(value.shape) for name, value in static_fields.items()}
    samples_by_year = {
        int(Path(record["path"]).stem): max(
            int(record["shape"][0]) - input_steps - output_steps + 1,
            0,
        )
        for record in generated
    }
    split_stats = {
        name: {
            "years": years,
            "samples": sum(samples_by_year.get(year, 0) for year in years),
        }
        for name, years in split_years.items()
    }
    card = {
        "name": "aurora-synthetic-era5",
        "version": "1.0.0",
        "created_at": now,
        "description": "Structured synthetic ERA5 fields for Aurora workflow validation only",
        "domain": "earth",
        "format": "OneScience ERA5Dataset HDF5",
        "files": [record["path"] for record in generated],
        "static": static_shapes,
        "usage": {"loader": "onescience.datapipes.climate.ERA5Dataset"},
    }
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
            "validate": f"python scripts/fake_data.py --config {config_path} --output-dir {output_dir} --validate-only",
        },
    }
    payloads = {
        "dataset_card.json": card,
        "statistics.json": {"files": generated, "splits": split_stats},
        "splits.json": split_stats,
        "lineage.json": lineage,
    }
    for name, payload in payloads.items():
        with (metadata_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def decode_variables(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def validate_h5(path: Path, expected_channels: list[str], expected_shape: tuple[int, int], time_step_hours: int) -> dict[str, Any]:
    """Validate one generated HDF5 file and its channel/statistics contract."""
    with h5py.File(path, "r") as handle:
        missing = sorted({"fields", "global_means", "global_stds"} - set(handle.keys()))
        if missing:
            raise ValueError(f"{path}: missing HDF5 keys {missing}")
        fields = handle["fields"]
        if decode_variables(fields.attrs["variables"]) != expected_channels:
            raise ValueError(f"{path}: channel order does not match configuration")
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
                raise ValueError(f"{path}: NaN or infinity at time index {step}")
            minimum = min(minimum, float(frame.min()))
            maximum = max(maximum, float(frame.max()))
        if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
            raise ValueError(f"{path}: invalid statistics")
        return {"path": str(path), "shape": list(fields.shape), "dtype": str(fields.dtype),
                "size_bytes": path.stat().st_size, "minimum": minimum, "maximum": maximum,
                "minimum_std": float(stds.min())}


def validate_static(path: Path, height: int, width: int) -> dict[str, Any]:
    with np.load(path) as static:
        required = {"lsm", "z", "slt", "lat", "lon"}
        missing = sorted(required - set(static.files))
        if missing:
            raise ValueError(f"{path}: missing static arrays {missing}")
        for name in ("lsm", "z", "slt"):
            if static[name].shape != (height, width) or not np.isfinite(static[name]).all():
                raise ValueError(f"{path}: invalid static field {name}")
        if static["lat"].shape != (height,) or static["lon"].shape != (width,):
            raise ValueError(f"{path}: coordinate shape mismatch")
        if not np.all(np.diff(static["lat"]) < 0) or not np.all(np.diff(static["lon"]) > 0):
            raise ValueError(f"{path}: coordinate order is incompatible with Aurora")
        return {name: list(static[name].shape) for name in required}


def validate_onescience_dataset(dataset_dir: Path, years: list[int], channels: list[str], input_steps: int,
                                output_steps: int, normalize: bool, height: int, width: int) -> dict[str, Any]:
    """Read one sample through OneScience ERA5Dataset and validate its shapes and values."""
    import torch
    from onescience.datapipes.climate import ERA5Dataset

    dataset = ERA5Dataset(dataset_dir=str(dataset_dir), used_years=years, used_variables=channels,
                          input_steps=input_steps, output_steps=output_steps, normalize=normalize)
    invar, outvar, cos_zenith, step_idx, time_index = dataset[0]
    expected_input = (input_steps, len(channels), height, width)
    expected_output = (len(channels), height, width) if output_steps == 1 else (output_steps, len(channels), height, width)
    if tuple(invar.shape) != expected_input or tuple(outvar.shape) != expected_output:
        raise ValueError(f"unexpected OneScience sample shapes: input={tuple(invar.shape)} output={tuple(outvar.shape)}")
    if not torch.isfinite(invar).all().item() or not torch.isfinite(outvar).all().item():
        raise ValueError("OneScience sample contains NaN or infinity")
    return {"length": len(dataset), "input_shape": list(invar.shape), "output_shape": list(outvar.shape),
            "cos_zenith_shape": list(cos_zenith.shape), "first_step_index": int(step_idx),
            "first_time_index": list(time_index), "years": years}


def validate_dataset(dataset_dir: Path, cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Validate generated files, static fields, and train/val/test loader samples."""
    data_cfg = cfg["data"]
    height, width = int(data_cfg["grid"]["virtual_height"]), int(data_cfg["grid"]["virtual_width"])
    channels = list(data_cfg["channel_order"])
    split_years = {"train": list(data_cfg["train_years"]), "val": list(data_cfg["val_years"]), "test": list(data_cfg["test_years"])}
    file_stats = []
    for year in sorted({year for years in split_years.values() for year in years}):
        path = dataset_dir / "data" / f"{year}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated year: {path}")
        file_stats.append(validate_h5(path, channels, (height, width), int(data_cfg["time_step_hours"])))
    static_stats = validate_static(dataset_dir / "static" / "static_vars.npz", height, width)
    split_stats = {}
    for name, years in split_years.items():
        split_stats[name] = validate_onescience_dataset(dataset_dir, years, channels, int(data_cfg["input_steps"]),
                                                        int(data_cfg["output_steps"]), bool(data_cfg["normalize_in_onescience"]),
                                                        height, width)
        print(f"validated {name}: {split_stats[name]}")
    summary = {"status": "validated", "dataset_dir": str(dataset_dir), "config": str(config_path),
               "files": file_stats, "static": static_stats, "splits": split_stats}
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--height", type=int, default=None, help="Override the configured grid height")
    parser.add_argument("--width", type=int, default=None, help="Override the configured grid width")
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing dataset without generating files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    grid_cfg = data_cfg["grid"]
    if args.height is not None:
        grid_cfg["virtual_height"] = args.height
    if args.width is not None:
        grid_cfg["virtual_width"] = args.width
    height = int(grid_cfg["virtual_height"])
    width = int(grid_cfg["virtual_width"])
    patch_size = int(cfg["model"]["patch_size"])
    if height < patch_size or height % patch_size not in {0, 1}:
        raise ValueError("Grid height must be divisible by patch_size or have exactly one extra row")
    if width < patch_size or width % patch_size != 0:
        raise ValueError("Grid width must be divisible by patch_size")
    output_dir = args.output_dir or resolve_project_path(data_cfg["virtual_dir"], config_path)
    output_dir = output_dir.resolve()
    if args.validate_only:
        validate_dataset(output_dir, cfg, config_path)
        print(f"dataset validation passed: {output_dir}")
        return
    years = args.years or sorted(
        set(data_cfg["train_years"] + data_cfg["val_years"] + data_cfg["test_years"])
    )
    timesteps = args.timesteps or int(data_cfg["virtual_timesteps"])
    channels = list(data_cfg["channel_order"])
    if len(channels) != 69 or len(channels) != len(set(channels)):
        raise ValueError("Aurora base-model channel_order must contain 69 unique channels")
    if timesteps < data_cfg["input_steps"] + data_cfg["output_steps"]:
        raise ValueError("Not enough timesteps for one input/output sample")

    output_dir.mkdir(parents=True, exist_ok=True)
    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    static_fields = make_static_fields(height, width)
    np.savez_compressed(static_dir / "static_vars.npz", **static_fields)

    generated = []
    for year in years:
        record = generate_year(
            output_dir / "data" / f"{year}.h5",
            channels,
            timesteps,
            height,
            width,
            int(data_cfg["time_step_hours"]),
            int(cfg["project"]["seed"]),
            year,
        )
        generated.append(record)
        print(f"generated {record['path']} shape={record['shape']}")

    generation = {
        "schema_version": "aurora-synthetic-era5-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "years": years,
        "channels": channels,
        "time_step_hours": int(data_cfg["time_step_hours"]),
        "static_file": str(static_dir / "static_vars.npz"),
        "files": generated,
    }
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "generation.json").open("w", encoding="utf-8") as handle:
        json.dump(generation, handle, indent=2)
    split_years = {
        "train": list(data_cfg["train_years"]),
        "val": list(data_cfg["val_years"]),
        "test": list(data_cfg["test_years"]),
    }
    write_dataset_metadata(
        output_dir,
        config_path,
        generated,
        split_years,
        static_fields,
        int(data_cfg["input_steps"]),
        int(data_cfg["output_steps"]),
    )
    validate_dataset(output_dir, cfg, config_path)
    print(f"generated metadata {metadata_dir / 'generation.json'}")


if __name__ == "__main__":
    main()
