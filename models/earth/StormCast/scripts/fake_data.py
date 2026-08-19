from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from grid import lambert_grid


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_temporal_fields(
    path: Path,
    variables: list[str],
    num_timesteps: int,
    image_size: tuple[int, int],
    time_step_hours: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = len(variables)
    height, width = image_size
    means = np.zeros((1, channels, 1, 1), dtype=np.float32)
    stds = np.ones((1, channels, 1, 1), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        fields = handle.create_dataset(
            "fields",
            shape=(num_timesteps, channels, height, width),
            dtype=np.float32,
            chunks=(1, channels, height, width),
            fillvalue=0.0,
        )
        fields.attrs["variables"] = variables
        fields.attrs["time_step"] = time_step_hours
        handle.create_dataset("global_means", data=means)
        handle.create_dataset("global_stds", data=stds)


def write_invariants(
    path: Path,
    variables: list[str],
    image_size: tuple[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_size
    target_lat, target_lon = lambert_grid(image_size)
    invariants = np.zeros((len(variables), height, width), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        fields = handle.create_dataset(
            "fields",
            shape=(len(variables), height, width),
            dtype=np.float32,
            chunks=(1, height, width),
            data=invariants,
        )
        fields.attrs["variables"] = variables
        handle.create_dataset("lat", data=target_lat)
        handle.create_dataset("lon", data=target_lon)


def generate(config: dict) -> None:
    data = config["data"]
    root = Path(data["root_dir"])
    years = sorted(
        set(data["train_years"] + data["val_years"] + data["test_years"])
    )
    era5_image_size = tuple(data["era5_image_size"])
    image_size = tuple(data["image_size"])
    if era5_image_size != (721, 1440):
        raise ValueError("ERA5 grid must be 721 x 1440")
    if image_size != (512, 640):
        raise ValueError("Regional grid must be 512 x 640")

    if len(data["era5_variables"]) != 26:
        raise ValueError("The configured ERA5 input must contain 26 channels")
    if len(data["state_variables"]) != 99:
        raise ValueError("The configured local state must contain 99 channels")
    if data["invariant_variables"] != ["lsm", "orography"]:
        raise ValueError("Invariant order must be [lsm, orography]")

    for year in years:
        write_temporal_fields(
            root / "era5" / "data" / f"{year}.h5",
            data["era5_variables"],
            data["num_timesteps"],
            era5_image_size,
            data["time_step_hours"],
        )
        write_temporal_fields(
            root / "hrrr" / "data" / f"{year}.h5",
            data["state_variables"],
            data["num_timesteps"],
            image_size,
            data["time_step_hours"],
        )

    write_invariants(
        root / "hrrr" / "invariants.h5",
        data["invariant_variables"],
        image_size,
    )
    print(f"Generated project validation data under {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lightweight StormCast data")
    parser.add_argument("--config", default="conf/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    project_root = config_path.parent.parent
    for key in ("root_dir",):
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str((project_root / path).resolve())
    generate(config)
