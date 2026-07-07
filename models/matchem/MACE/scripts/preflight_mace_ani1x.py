#!/usr/bin/env python3
"""Preflight check for the standardized MACE + ANI-1x runtime package."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import yaml


REQUIRED_H5_KEYS = (
    "atomic_numbers",
    "positions",
    "properties/energy",
    "properties/forces",
)


def expand_path(value: str, package_root: Path, data_dir: Path) -> Path:
    expanded = os.path.expandvars(value)
    expanded = expanded.replace("${MACE_ANI1X_DATA_DIR:-data/ani1x}", str(data_dir))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return package_root / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"[FAIL] YAML 不是对象: {path}")
    return data


def check_h5_sample(path: Path) -> None:
    import h5py

    with h5py.File(path, "r") as f:
        batches = sorted(f.keys())
        if not batches:
            raise SystemExit(f"[FAIL] HDF5 无 config_batch: {path}")
        configs = sorted(f[batches[0]].keys())
        if not configs:
            raise SystemExit(f"[FAIL] HDF5 batch 中无 config: {path}")
        sample = f[batches[0]][configs[0]]
        for key in REQUIRED_H5_KEYS:
            if key not in sample:
                raise SystemExit(f"[FAIL] HDF5 样本缺少字段 {key}: {path}")
        forces_shape = sample["properties/forces"].shape
        positions_shape = sample["positions"].shape
        if len(positions_shape) != 2 or positions_shape[1] != 3:
            raise SystemExit(f"[FAIL] positions shape 异常: {path} {positions_shape}")
        if forces_shape != positions_shape:
            raise SystemExit(f"[FAIL] forces 与 positions shape 不一致: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ani1x_8dcu.yaml")
    parser.add_argument("--data-dir", default="data/ani1x")
    args = parser.parse_args()

    package_root = Path.cwd()
    config_path = package_root / args.config
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = package_root / data_dir

    cfg = load_yaml(config_path)
    train_args = cfg.get("train_args") or {}
    if not isinstance(train_args, dict):
        raise SystemExit("[FAIL] config.train_args 缺失或格式错误")

    required_paths = {
        "train_file": train_args.get("train_file"),
        "valid_file": train_args.get("valid_file"),
        "test_file": train_args.get("test_file"),
        "statistics_file": train_args.get("statistics_file"),
    }
    for key, raw_value in required_paths.items():
        if not raw_value:
            raise SystemExit(f"[FAIL] config.train_args.{key} 缺失")
        resolved = expand_path(str(raw_value), package_root, data_dir)
        if not resolved.exists():
            raise SystemExit(f"[FAIL] {key} 不存在: {resolved}")
        print(f"[OK] {key}: {resolved}")

    stats_path = expand_path(str(required_paths["statistics_file"]), package_root, data_dir)
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("atomic_energies", "avg_num_neighbors", "mean", "std", "atomic_numbers", "r_max"):
        if key not in stats:
            raise SystemExit(f"[FAIL] statistics 缺少字段: {key}")
    if float(train_args.get("r_max")) != float(stats["r_max"]):
        raise SystemExit(
            f"[FAIL] config.train_args.r_max={train_args.get('r_max')} "
            f"与 statistics.r_max={stats['r_max']} 不一致"
        )
    if str(train_args.get("E0s")) != str(stats["atomic_energies"]):
        raise SystemExit("[FAIL] config.train_args.E0s 与 statistics.atomic_energies 不一致")
    print("[OK] statistics 与配置匹配")

    for key in ("train_file", "valid_file", "test_file"):
        folder = expand_path(str(required_paths[key]), package_root, data_dir)
        h5_files = sorted(glob.glob(str(folder / "*.h5")))
        if not h5_files:
            raise SystemExit(f"[FAIL] {key} 目录无 h5 文件: {folder}")
        check_h5_sample(Path(h5_files[0]))
        print(f"[OK] {key}: {len(h5_files)} 个 h5 分片，样本结构可读")

    print("[OK] MACE ANI-1x 标准运行包预检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
