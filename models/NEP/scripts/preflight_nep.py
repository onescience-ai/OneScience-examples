#!/usr/bin/env python3
"""Preflight checks for the standardized NEP runtime package."""

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
    if cfg.get("model_type") != "NEP":
        raise ValueError(f"model_type must be NEP, got {cfg.get('model_type')!r}")
    if cfg.get("format") != "pwmat/movement":
        raise ValueError(f"standard Cu config format must be pwmat/movement, got {cfg.get('format')!r}")

    expected_train = [
        "data/MatPL/Cu/pwdata/0_300_MOVEMENT",
        "data/MatPL/Cu/pwdata/1_500_MOVEMENT",
    ]
    expected_valid = ["data/MatPL/Cu/pwdata/valid_movement"]
    if cfg.get("train_data") != expected_train:
        raise ValueError(f"train_data mismatch: {cfg.get('train_data')!r}")
    if cfg.get("valid_data") != expected_valid:
        raise ValueError(f"valid_data mismatch: {cfg.get('valid_data')!r}")

    for rel in expected_train + expected_valid:
        path = (dataset_root / rel.removeprefix("data/MatPL/")) if dataset_root else (package_root / rel)
        require_file(path)
    print("[OK] NEP config matches standardized MatPL Cu data layout")


def check_runtime_files(package_root: Path) -> None:
    for rel in [
        "upstream/matpl_install.sh",
        "upstream/nep_README.md",
        "upstream/demo/nep_Cu/submit.sh",
        "conf/Cu_nep_train_modelscope.json",
    ]:
        require_file(package_root / rel)
    print("[OK] NEP runtime files are present")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--config", default="conf/Cu_nep_train_modelscope.json")
    parser.add_argument("--dataset-root", default=None, help="Optional MatPL data root for separated local validation")
    args = parser.parse_args()

    package_root = Path(args.package_root).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    check_runtime_files(package_root)
    check_config(package_root / args.config, package_root, dataset_root)
    print("[OK] NEP model preflight passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"NEP model preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
