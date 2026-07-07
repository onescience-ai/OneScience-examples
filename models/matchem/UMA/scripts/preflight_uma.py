#!/usr/bin/env python3
"""Preflight checks for the standardized UMA runtime package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def require_path(path: Path, kind: str) -> None:
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"missing directory: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")


def read_yaml(path: Path) -> dict:
    require_path(path, "file")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def check_uma_config(config_path: Path, package_root: Path, dataset_root: Path | None, require_checkpoint: bool) -> None:
    cfg = read_yaml(config_path)
    data = cfg.get("data", {})
    if data.get("dataset_name") != "oc20":
        raise ValueError(f"data.dataset_name must be oc20, got {data.get('dataset_name')!r}")

    train_src = data.get("train_dataset", {}).get("splits", {}).get("train", {}).get("src")
    val_src = data.get("val_dataset", {}).get("splits", {}).get("val", {}).get("src")
    if train_src != "data/oc20/uma_oc20_finetune/train":
        raise ValueError(f"train src mismatch: {train_src!r}")
    if val_src != "data/oc20/uma_oc20_finetune/val":
        raise ValueError(f"val src mismatch: {val_src!r}")

    if dataset_root is None:
        train_dir = package_root / train_src
        val_dir = package_root / val_src
    else:
        train_dir = dataset_root / "uma_oc20_finetune/train"
        val_dir = dataset_root / "uma_oc20_finetune/val"
    require_path(train_dir, "dir")
    require_path(val_dir, "dir")

    checkpoint = (
        cfg.get("runner", {})
        .get("train_eval_unit", {})
        .get("model", {})
        .get("checkpoint_location")
    )
    if checkpoint != "checkpoints/uma-s-1p1_converted.pt":
        raise ValueError(f"checkpoint_location mismatch: {checkpoint!r}")
    checkpoint_path = package_root / checkpoint
    if require_checkpoint:
        require_path(checkpoint_path, "file")
    elif not checkpoint_path.exists():
        print(f"[WARN] checkpoint not present: {checkpoint_path}")

    print("[OK] UMA config paths match standardized OC20 layout")


def check_runtime_files(package_root: Path) -> None:
    required_files = [
        "upstream/train.py",
        "upstream/demo/run.sh",
        "upstream/demo/_parse_config.py",
        "upstream/demo/templates/preflight_check.sh",
        "upstream/models/pretrained_models.json",
        "conf/oc20_ef_4dcu_modelscope.yaml",
    ]
    for rel in required_files:
        require_path(package_root / rel, "file")
    print("[OK] UMA runtime files are present")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", default=".", help="Standardized UMA package root")
    parser.add_argument("--config", default="conf/oc20_ef_4dcu_modelscope.yaml")
    parser.add_argument("--dataset-root", default=None, help="Optional OC20 data root for separated local validation")
    parser.add_argument("--require-checkpoint", action="store_true")
    args = parser.parse_args()

    package_root = Path(args.package_root).resolve()
    config_path = package_root / args.config
    check_runtime_files(package_root)
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    check_uma_config(config_path, package_root, dataset_root, args.require_checkpoint)
    print("[OK] UMA model preflight passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"UMA model preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
