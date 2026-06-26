#!/usr/bin/env python3
"""Preflight checks for the standardized DeePMD runtime package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing directory: {path}")


def load_json(path: Path) -> dict:
    require_file(path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def check_config(config_path: Path, package_root: Path, dataset_root: Path | None) -> None:
    cfg = load_json(config_path)
    if cfg.get("model", {}).get("descriptor", {}).get("type") != "se_e2_a":
        raise ValueError("standard config must use se_e2_a descriptor")
    if cfg.get("model", {}).get("type_map") != ["O", "H"]:
        raise ValueError(f"type_map mismatch: {cfg.get('model', {}).get('type_map')!r}")

    train_systems = cfg.get("training", {}).get("training_data", {}).get("systems")
    valid_systems = cfg.get("training", {}).get("validation_data", {}).get("systems")
    expected_train = ["data/DeePMD/water/data_0", "data/DeePMD/water/data_1", "data/DeePMD/water/data_2"]
    expected_valid = ["data/DeePMD/water/data_3"]
    if train_systems != expected_train:
        raise ValueError(f"training systems mismatch: {train_systems!r}")
    if valid_systems != expected_valid:
        raise ValueError(f"validation systems mismatch: {valid_systems!r}")

    for rel in expected_train + expected_valid:
        path = (dataset_root / rel.removeprefix("data/DeePMD/")) if dataset_root else (package_root / rel)
        require_dir(path)
        require_file(path / "type.raw")
        require_file(path / "type_map.raw")
        sets = sorted(path.glob("set.*"))
        if not sets:
            raise FileNotFoundError(f"missing set.* directory under {path}")
    print("[OK] DeePMD config matches standardized water data layout")


def check_runtime_files(package_root: Path) -> None:
    for rel in [
        "upstream/dp_install.sh",
        "upstream/dp_README.md",
        "upstream/demo/water_se_e2_a_pt/submit_1card.sh",
        "conf/input_torch_modelscope.json",
    ]:
        require_file(package_root / rel)
    print("[OK] DeePMD runtime files are present")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--config", default="conf/input_torch_modelscope.json")
    parser.add_argument("--dataset-root", default=None, help="Optional DeePMD data root for separated local validation")
    args = parser.parse_args()

    package_root = Path(args.package_root).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    check_runtime_files(package_root)
    check_config(package_root / args.config, package_root, dataset_root)
    print("[OK] DeePMD model preflight passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DeePMD model preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
