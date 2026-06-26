#!/usr/bin/env python3
"""Preflight checks for the XiHe ModelScope runtime package."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml


REQUIRED_YEARS = [1993, 1994, 1995, 1996, 1997, 1998, 1999]
REQUIRED_FIELD_SHAPE = (3, 96, 2041, 4320)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_attr_strings(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check XiHe config and CMEMS runtime data.")
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    dataset_cfg = cfg["datapipe"]["dataset"]

    data_dir = Path(args.data_dir or dataset_cfg["data_dir"]).resolve()
    h5_dir = data_dir / "data"
    stats_dir = Path(dataset_cfg["stats_dir"]).resolve()
    mask_path = Path(model_cfg["mask"]).resolve()

    expected_train = [1993, 1994, 1995, 1996, 1997]
    expected_val = [1998]
    expected_test = [1999]
    if dataset_cfg["train_time"] != expected_train:
        raise AssertionError(f"train_time mismatch: {dataset_cfg['train_time']} != {expected_train}")
    if dataset_cfg["val_time"] != expected_val:
        raise AssertionError(f"val_time mismatch: {dataset_cfg['val_time']} != {expected_val}")
    if dataset_cfg["test_time"] != expected_test:
        raise AssertionError(f"test_time mismatch: {dataset_cfg['test_time']} != {expected_test}")

    channels = list(dataset_cfg["channels"])
    if len(channels) != 96:
        raise AssertionError(f"config channels must contain 96 variables, got {len(channels)}")

    for year in REQUIRED_YEARS:
        path = h5_dir / f"{year}.h5"
        if not path.exists():
            raise FileNotFoundError(f"missing yearly HDF5 file: {path}")
        with h5py.File(path, "r") as handle:
            for key in ("fields", "global_means", "global_stds"):
                if key not in handle:
                    raise KeyError(f"{path} missing dataset {key}")
            fields = handle["fields"]
            if tuple(fields.shape) != REQUIRED_FIELD_SHAPE:
                raise AssertionError(f"{path} fields shape {fields.shape} != {REQUIRED_FIELD_SHAPE}")
            if fields.dtype != np.dtype("float32"):
                raise AssertionError(f"{path} fields dtype {fields.dtype} != float32")
            variables = normalize_attr_strings(fields.attrs["variables"])
            missing_channels = sorted(set(channels) - set(variables))
            if missing_channels:
                raise AssertionError(f"{path} missing configured channels: {missing_channels}")
            if int(fields.attrs["time_step"]) != int(dataset_cfg["time_res"]):
                raise AssertionError(f"{path} time_step does not match config time_res")

    for stat_name in ("global_means.npy", "global_stds.npy"):
        stat_path = stats_dir / stat_name
        if not stat_path.exists():
            raise FileNotFoundError(f"missing extracted stats file: {stat_path}")
        stat = np.load(stat_path)
        if tuple(stat.shape) != (1, 96, 1, 1):
            raise AssertionError(f"{stat_path} shape {stat.shape} != (1, 96, 1, 1)")

    if not mask_path.exists():
        raise FileNotFoundError(f"missing land mask: {mask_path}")
    mask = np.load(mask_path)
    if tuple(mask.shape) != REQUIRED_FIELD_SHAPE[-2:]:
        raise AssertionError(f"{mask_path} shape {mask.shape} != {REQUIRED_FIELD_SHAPE[-2:]}")

    print("XiHe preflight passed.")


if __name__ == "__main__":
    main()
