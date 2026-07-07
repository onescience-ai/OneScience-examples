#!/usr/bin/env python3
"""Preflight checks for the standardized ESMFold ModelScope package."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required to parse config/manifest: {exc}")


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def parse_fasta(path: Path) -> tuple[int, int]:
    records = 0
    residues = 0
    current = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    records += 1
                    residues += sum(len(part) for part in current)
                    current = []
            else:
                if set(line.upper()) - set("ACDEFGHIKLMNPQRSTVWYBXZUOJ-."):
                    raise ValueError(f"Invalid FASTA residue in line: {line[:80]}")
                current.append(line)
    if current:
        records += 1
        residues += sum(len(part) for part in current)
    return records, residues


def resolve_model_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT, help="model package root")
    parser.add_argument("--dataset-root", type=Path, default=None, help="downloaded dataset package root")
    parser.add_argument("--skip-sha256", action="store_true", help="skip full SHA256 checks for large files")
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = root / "conf" / "config.yaml"
    manifest_path = root / "manifest.yaml"
    config = read_yaml(config_path)
    manifest = read_yaml(manifest_path)

    errors: list[str] = []
    if config.get("resource_ids", {}).get("model") != "OneScience/esmfold/":
        errors.append("conf/config.yaml resource_ids.model must be OneScience/esmfold/")
    if config.get("resource_ids", {}).get("dataset") != "OneScience/esmfold_dataset":
        errors.append("conf/config.yaml resource_ids.dataset must be OneScience/esmfold_dataset")
    if manifest.get("resource", {}).get("id") != "OneScience/esmfold/":
        errors.append("manifest resource.id must be OneScience/esmfold/")

    sample_fasta = resolve_model_path(config["paths"]["sample_fasta"], root)
    checkpoint = resolve_model_path(config["paths"]["checkpoint"], root)
    for required in [sample_fasta, checkpoint]:
        if not required.exists():
            errors.append(f"Missing required file: {required}")
        elif required.stat().st_size <= 0:
            errors.append(f"Required file is empty: {required}")

    if sample_fasta.exists():
        records, residues = parse_fasta(sample_fasta)
        if records < 1:
            errors.append(f"No FASTA records found in {sample_fasta}")
        print(f"FASTA OK: {sample_fasta} records={records} residues={residues}")

    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = (root / config["dataset_expectation"]["expected_dataset_local_path"]).resolve()
    dataset_weight = dataset_root / "weight" / "esmfold_3B_v1.pt"
    if not dataset_weight.exists():
        errors.append(f"Dataset weight not found: {dataset_weight}")
    elif checkpoint.exists() and checkpoint.stat().st_size != dataset_weight.stat().st_size:
        errors.append("Model checkpoint and dataset esmfold_3B_v1.pt sizes differ")

    expected = {
        "checkpoints/esmfold_3B_v1.pt": 2771653574,
        "data/sample/few_proteins.fasta": 319,
    }
    for rel_path, size in expected.items():
        path = root / rel_path
        if path.exists() and path.stat().st_size != size:
            errors.append(f"Unexpected size for {rel_path}: {path.stat().st_size} != {size}")

    if not args.skip_sha256:
        expected_sha256 = {
            "data/sample/few_proteins.fasta": "910b90110764f0abe23d1378ef4e8bb721917bcb9fa6f2749ef06dcaea3f6e0a",
            "checkpoints/esmfold_3B_v1.pt": "e9a52579027e77d2d2e0a18218e755821f395730e86624cab9413dc117f5ca62",
        }
        for rel_path, expected_digest in expected_sha256.items():
            path = root / rel_path
            if path.exists() and expected_digest != "TO_BE_FILLED":
                actual = sha256_file(path)
                if actual != expected_digest:
                    errors.append(f"SHA256 mismatch for {rel_path}: {actual} != {expected_digest}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Model preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
