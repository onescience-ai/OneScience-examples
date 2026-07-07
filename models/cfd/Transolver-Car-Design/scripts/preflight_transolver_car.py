#!/usr/bin/env python3
"""Preflight checks for the Transolver-Car-Design runtime package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


REQUIRED_STATS = {
    "mean_in.npy": ((7,), np.floating),
    "std_in.npy": ((7,), np.floating),
    "mean_out.npy": ((4,), np.floating),
    "std_out.npy": ((4,), np.floating),
}
PREPROCESSED_FILES = {
    "x.npy": 2,
    "y.npy": 2,
    "pos.npy": 2,
    "surf.npy": 1,
    "edge_index.npy": 2,
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        fail(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        fail(f"config is not a YAML mapping: {path}")
    return data


def check_stats(stats_dir: Path) -> None:
    if not stats_dir.is_dir():
        fail(f"stats_dir not found: {stats_dir}")
    for filename, (shape, dtype_kind) in REQUIRED_STATS.items():
        path = stats_dir / filename
        if not path.is_file():
            fail(f"missing stats file: {path}")
        arr = np.load(path, allow_pickle=False)
        if arr.shape != shape:
            fail(f"{path} shape mismatch: expected {shape}, got {arr.shape}")
        if not np.issubdtype(arr.dtype, dtype_kind):
            fail(f"{path} dtype mismatch: expected floating, got {arr.dtype}")


def check_training_dir(training_dir: Path) -> None:
    if not training_dir.is_dir():
        fail(f"training data_dir not found: {training_dir}")
    param_dirs = sorted(p for p in training_dir.glob("param*") if p.is_dir())
    if not param_dirs:
        fail(f"no param* directories under {training_dir}")
    for param_dir in param_dirs:
        for filename in ("Cd.npy", "I1.npy", "I2.npy", "Press.npy", "Velo.npy"):
            if not (param_dir / filename).is_file():
                fail(f"missing training file: {param_dir / filename}")


def check_preprocessed_dir(preprocessed_dir: Path) -> None:
    if not preprocessed_dir.is_dir():
        fail(f"preprocessed_save_dir not found: {preprocessed_dir}")
    sample_dirs = sorted(p for p in preprocessed_dir.glob("param*/*") if p.is_dir())
    if not sample_dirs:
        fail(f"no preprocessed sample directories under {preprocessed_dir}")
    sample = sample_dirs[0]
    for filename, ndim in PREPROCESSED_FILES.items():
        path = sample / filename
        if not path.is_file():
            fail(f"missing preprocessed sample file: {path}")
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != ndim:
            fail(f"{path} ndim mismatch: expected {ndim}, got {arr.ndim}")
    x = np.load(sample / "x.npy", allow_pickle=False)
    y = np.load(sample / "y.npy", allow_pickle=False)
    pos = np.load(sample / "pos.npy", allow_pickle=False)
    surf = np.load(sample / "surf.npy", allow_pickle=False)
    edge_index = np.load(sample / "edge_index.npy", allow_pickle=False)
    if x.shape[1] != 7:
        fail(f"{sample / 'x.npy'} feature dimension should be 7, got {x.shape}")
    if y.shape[1] != 4:
        fail(f"{sample / 'y.npy'} target dimension should be 4, got {y.shape}")
    if pos.shape[1] != 3:
        fail(f"{sample / 'pos.npy'} coordinate dimension should be 3, got {pos.shape}")
    if surf.shape[0] != x.shape[0] or y.shape[0] != x.shape[0] or pos.shape[0] != x.shape[0]:
        fail(f"sample node count mismatch in {sample}")
    if edge_index.shape[0] != 2:
        fail(f"{sample / 'edge_index.npy'} should have shape [2, num_edges], got {edge_index.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/transolver_car.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    root = config_path.resolve().parent.parent
    config = load_yaml(config_path)
    source = config.get("datapipe", {}).get("source", {})
    data_dir = root / source.get("data_dir", "")
    preprocessed_dir = root / source.get("preprocessed_save_dir", "")
    stats_dir = root / source.get("stats_dir", "")

    check_training_dir(data_dir)
    check_preprocessed_dir(preprocessed_dir)
    check_stats(stats_dir)

    model_name = config.get("model", {}).get("name")
    specific_params = config.get("model", {}).get("specific_params", {})
    if model_name not in specific_params:
        fail(f"model.name {model_name!r} not found in model.specific_params")
    params = specific_params[model_name]
    if params.get("space_dim") != 7:
        fail("model space_dim must match x.npy feature dimension 7")
    if params.get("out_dim") != 4:
        fail("model out_dim must match y.npy target dimension 4")

    print("[OK] Transolver-Car-Design preflight passed")


if __name__ == "__main__":
    main()
