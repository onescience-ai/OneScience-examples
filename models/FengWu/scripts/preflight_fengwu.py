#!/usr/bin/env python3
"""Preflight checks for the standardized FengWu runtime package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import yaml


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_data_dir(workdir: Path, config: dict) -> Path:
    data_dir = Path(config["datapipe"]["dataset"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = workdir / data_dir
    return data_dir


def _years(config: dict) -> list[int]:
    dataset = config["datapipe"]["dataset"]
    years = []
    for key in ("train_time", "val_time", "test_time"):
        years.extend(int(y) for y in dataset[key])
    return sorted(set(years))


def check_h5(path: Path, channels: list[str], img_size: list[int]) -> None:
    with h5py.File(path, "r") as f:
        for name in ("fields", "global_means", "global_stds"):
            if name not in f:
                raise RuntimeError(f"{path}: missing dataset {name}")
        fields = f["fields"]
        if fields.dtype.name != "float32":
            raise RuntimeError(f"{path}: fields dtype is {fields.dtype}, expected float32")
        if len(fields.shape) != 4:
            raise RuntimeError(f"{path}: fields shape is {fields.shape}, expected [T, C, H, W]")
        if list(fields.shape[2:]) != list(img_size):
            raise RuntimeError(f"{path}: spatial shape {fields.shape[2:]} != {img_size}")
        variables = [
            v.decode() if isinstance(v, bytes) else str(v)
            for v in fields.attrs.get("variables", [])
        ]
        missing = [c for c in channels if c not in variables]
        if missing:
            raise RuntimeError(f"{path}: missing configured channels: {missing[:10]}")
        if int(fields.attrs.get("time_step", -1)) <= 0:
            raise RuntimeError(f"{path}: missing or invalid fields.attrs['time_step']")
        if f["global_means"].shape[1] != len(variables):
            raise RuntimeError(f"{path}: global_means channel count does not match variables")
        if f["global_stds"].shape[1] != len(variables):
            raise RuntimeError(f"{path}: global_stds channel count does not match variables")
        _ = fields[0, 0, 0, 0]
        _ = f["global_means"][0, 0, 0, 0]
        _ = f["global_stds"][0, 0, 0, 0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=".", help="standardized model package root")
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--require-checkpoint", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    config_path = workdir / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {config_path}")

    config = _load_config(config_path)
    dataset_cfg = config["datapipe"]["dataset"]
    data_dir = _resolve_data_dir(workdir, config)
    h5_dir = data_dir / "data"
    if not h5_dir.is_dir():
        raise FileNotFoundError(
            f"missing ERA5 HDF5 directory: {h5_dir}; download OneScience/ERA5 to {data_dir}"
        )

    years = _years(config)
    missing_files = [str(h5_dir / f"{year}.h5") for year in years if not (h5_dir / f"{year}.h5").exists()]
    if missing_files:
        raise FileNotFoundError(f"missing configured year files: {missing_files[:10]}")

    channels = list(dataset_cfg["channels"])
    if len(channels) != 189:
        raise RuntimeError(f"FengWu config expects 189 channels, got {len(channels)}")
    if int(config["model"]["pressure_level"]) != 37:
        raise RuntimeError("model.pressure_level must be 37 for the configured FengWu channel groups")

    first_file = h5_dir / f"{years[0]}.h5"
    last_file = h5_dir / f"{years[-1]}.h5"
    check_h5(first_file, channels, list(dataset_cfg["img_size"]))
    if last_file != first_file:
        check_h5(last_file, channels, list(dataset_cfg["img_size"]))

    checkpoint_dir = Path(config["model"]["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = workdir / checkpoint_dir
    if args.require_checkpoint and not (checkpoint_dir / "model_bak.pth").exists():
        raise FileNotFoundError(f"missing checkpoint for inference/evaluate: {checkpoint_dir / 'model_bak.pth'}")

    print("FengWu preflight passed")
    print(f"config: {config_path}")
    print(f"data_dir: {data_dir}")
    print(f"years: {years[0]}-{years[-1]} ({len(years)} files)")
    print(f"channels: {len(channels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
