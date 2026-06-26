#!/usr/bin/env python3
"""Lightweight AlphaGenome package preflight."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


REQUIRED_ENTRYPOINTS = [
    "run_inference.py",
    "run_variant_scoring.py",
    "run_track_prediction_eval.py",
]

REQUIRED_CHECKPOINT_PATHS = [
    "_CHECKPOINT_METADATA",
    "_METADATA",
    "manifest.ocdbt",
    "ocdbt.process_0/manifest.ocdbt",
]

OPTIONAL_IMPORTS = [
    "absl",
    "jax",
    "numpy",
    "orbax.checkpoint",
    "pandas",
    "tensorflow",
    "onescience",
]


def module_available(name: str) -> bool:
    parts = name.split(".")
    spec = importlib.util.find_spec(parts[0])
    if spec is None:
        return False
    if len(parts) == 1:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="checkpoints/alphagenome-all-folds",
        help="Path to the local AlphaGenome Orbax/OCDBT checkpoint.",
    )
    parser.add_argument("--fasta_path", default=None)
    parser.add_argument("--vcf_path", default=None)
    parser.add_argument("--eval_data_dir", default=None)
    args = parser.parse_args()

    root = Path.cwd()
    failures: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_ENTRYPOINTS:
        if not (root / rel).is_file():
            failures.append(f"missing entrypoint: {rel}")

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    if not model_dir.is_dir():
        failures.append(f"missing model_dir: {model_dir}")
    else:
        for rel in REQUIRED_CHECKPOINT_PATHS:
            if not (model_dir / rel).exists():
                failures.append(f"missing checkpoint path: {model_dir / rel}")
        data_files = [p for p in (model_dir / "ocdbt.process_0" / "d").glob("*") if p.is_file()]
        if len(data_files) < 3:
            failures.append(
                "checkpoint data shards look incomplete: "
                f"{model_dir / 'ocdbt.process_0' / 'd'}"
            )

    for optional_path, label in [
        (args.fasta_path, "fasta_path"),
        (args.vcf_path, "vcf_path"),
        (args.eval_data_dir, "eval_data_dir"),
    ]:
        if optional_path and not Path(optional_path).exists():
            failures.append(f"{label} does not exist: {optional_path}")

    if args.fasta_path and not Path(args.fasta_path + ".fai").exists():
        warnings.append(f"FASTA index not found: {args.fasta_path}.fai")

    missing_imports = [name for name in OPTIONAL_IMPORTS if not module_available(name)]
    if missing_imports:
        warnings.append("optional imports missing: " + ", ".join(missing_imports))

    for warning in warnings:
        print("WARNING:", warning)

    if failures:
        for failure in failures:
            print("ERROR:", failure, file=sys.stderr)
        return 1

    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
