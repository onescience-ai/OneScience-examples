#!/usr/bin/env python3
"""Preflight checks for the ModelScope AlphaGenome runtime package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import sys

import yaml


REQUIRED_CHECKPOINT_FILES = [
    "_CHECKPOINT_METADATA",
    "_METADATA",
    "manifest.ocdbt",
    "ocdbt.process_0/manifest.ocdbt",
]

REQUIRED_REFERENCE_FILES = [
    "reference/HOMO_SAPIENS/GRCh38.p13.genome.fa",
    "reference/HOMO_SAPIENS/GRCh38.p13.genome.fa.fai",
]

REQUIRED_DATA_PATTERNS = [
    "v1/train/ALL_FOLDS/HOMO_SAPIENS/VALID/ATAC/data_chrAll_01-25.gz.tfrecord",
    "v1/train/ALL_FOLDS/HOMO_SAPIENS/VALID/DNASE/data_chrAll_01-25.gz.tfrecord",
    "v1/train/ALL_FOLDS/HOMO_SAPIENS/VALID/RNA_SEQ/data_chrAll_01-25.gz.tfrecord",
]


def sha256_head(path: Path, size: int = 1048576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        digest.update(f.read(size))
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="model package root")
    parser.add_argument(
        "--dataset-root",
        default="data/alphagenome_dataset",
        help="dataset root relative to --root or absolute",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/alphagenome-all-folds",
        help="checkpoint dir relative to --root or absolute",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config_path = root / "conf" / "alphagenome_paths.yaml"
    require_file(config_path, "config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if config["model"]["repo_id"] != "OneScience/alphagenome/":
        raise ValueError("model repo_id mismatch")
    if config["dataset"]["repo_id"] != "OneScience/alphagenome_dataset":
        raise ValueError("dataset repo_id mismatch")

    checkpoint = Path(args.checkpoint_dir)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    for rel in REQUIRED_CHECKPOINT_FILES:
        require_file(checkpoint / rel, "checkpoint file")

    data_root = Path(args.dataset_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    for rel in REQUIRED_REFERENCE_FILES + REQUIRED_DATA_PATTERNS:
        require_file(data_root / rel, "dataset file")

    for rel in REQUIRED_DATA_PATTERNS:
        with gzip.open(data_root / rel, "rb") as f:
            if not f.read(1):
                raise ValueError(f"gzip TFRecord is unreadable: {data_root / rel}")

    print("preflight_ok: true")
    print(f"checkpoint_dir: {checkpoint}")
    print(f"dataset_root: {data_root}")
    print(f"reference_head_sha256: {sha256_head(data_root / REQUIRED_REFERENCE_FILES[0])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"preflight_ok: false\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1)
