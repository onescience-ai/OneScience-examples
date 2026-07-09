#!/usr/bin/env python
"""Preflight checks for the standalone RFdiffusion package."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DGLBACKEND", "pytorch")
REQUIRED_WEIGHTS = [
    "ActiveSite_ckpt.pt",
    "Base_ckpt.pt",
    "Base_epoch8_ckpt.pt",
    "Complex_Fold_base_ckpt.pt",
    "Complex_base_ckpt.pt",
    "Complex_beta_ckpt.pt",
    "InpaintSeq_Fold_ckpt.pt",
    "InpaintSeq_ckpt.pt",
    "RF_structure_prediction_weights.pt",
]
REQUIRED_FILES = [
    "README.md",
    "configuration.json",
    "config/inference/base.yaml",
    "config/inference/symmetry.yaml",
    "scripts/run_inference.py",
    "examples/input_pdbs/1qys.pdb",
    "examples/input_pdbs/1YCR.pdb",
]
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".sh", ".txt", ".tsv"}
GENERATED_DIRS = {"outputs", ".cache", "logs"}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def check_required_files() -> None:
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).is_file()]
    if missing:
        fail("Missing required files: " + ", ".join(missing))
    print(f"[OK] Required files present: {len(REQUIRED_FILES)}")


def check_weights(strict: bool) -> None:
    missing = []
    bad = []
    for name in REQUIRED_WEIGHTS:
        path = ROOT / "weight" / name
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
            continue
        if strict:
            size = path.stat().st_size
            head = path.read_bytes()[:128]
            lfs_marker = b"version https://git-lfs" + b".github.com"
            if size < 1024 * 1024 or head.startswith(lfs_marker):
                bad.append(f"{path.relative_to(ROOT)} ({size} bytes)")
    if missing:
        fail("Missing weight files: " + ", ".join(missing))
    if bad:
        fail("Invalid or placeholder weight files: " + ", ".join(bad))
    mode = "strict" if strict else "basic"
    print(f"[OK] Weight files present ({mode}): {len(REQUIRED_WEIGHTS)}")


def iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if GENERATED_DIRS.intersection(path.relative_to(ROOT).parts):
            continue
        if path.name == ".gitattributes" or path.suffix in TEXT_SUFFIXES:
            yield path



def check_imports() -> None:
    sys.path.insert(0, str(ROOT))
    modules = [
        "onescience.utils.rfdiffusion.inference.utils",
        "onescience.utils.rfdiffusion.inference.model_runners",
        "onescience.models.rfdiffusion.RoseTTAFoldModel",
        "onescience.models.se3_transformer",
    ]
    for name in modules:
        importlib.import_module(name)
    print(f"[OK] Strict imports succeeded: {len(modules)} modules")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-weights", action="store_true")
    parser.add_argument("--strict-imports", action="store_true")
    args = parser.parse_args()

    check_required_files()
    check_weights(strict=args.strict_weights)
    if args.strict_imports:
        check_imports()
    print("[OK] RFdiffusion standalone preflight passed")


if __name__ == "__main__":
    main()
