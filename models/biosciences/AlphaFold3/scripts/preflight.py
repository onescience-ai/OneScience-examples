#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required for preflight: pip install pyyaml") from exc


MODEL_FILES = {
    "af3.bin": (1146811260, "a43670ea1fae790cd30ccc8c8bf836c4098014c45250fb310a5338407e321de7"),
    "libflash_atten_c.so": (75882104, "2ea956e8d13317dd795ffce846079f20e502a39df66802f485d5f25900ef493b"),
    "mmseqs/bin/mmseqs": (14417192, "d166c67ca79089ea5f89ec1133f67ab1c216e1734a0408bbc7205e588fe865ed"),
    "mmseqs/lib/libmarv.so": (58277568, "fdc787467b6d05d36db1f034a2cb8944b5b072bbe6d46172013810a066583928"),
}

DATASET_REQUIRED = [
    "infer_input_data/readme.md",
    "infer_input_data/all_data/7r6r_data.json",
    "infer_input_data/all_data/t1119_data.json",
    "public_databases/pdb_seqres_2022_09_28.fasta",
    "mmseqsDB/small_bfd_db",
    "mmseqsDB/small_bfd_db.dbtype",
    "jackhmmer_split/bfd-first_non_consensus_sequences.fasta-00000-of-00064",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("dialect") != "alphafold3":
        raise AssertionError(f"{path} dialect is not alphafold3")
    if not data.get("name"):
        raise AssertionError(f"{path} missing name")
    if not isinstance(data.get("sequences"), list) or not data["sequences"]:
        raise AssertionError(f"{path} missing sequences")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset-root", default="data/alphafold3_dataset")
    parser.add_argument("--model-dir", default="checkpoints/AlphaFold3")
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_dir = (root / args.model_dir).resolve()
    dataset_root = (root / args.dataset_root).resolve()

    config = yaml.safe_load((root / "conf" / "alphafold3_paths.yaml").read_text(encoding="utf-8"))
    if config["model"]["repo_id"] != "OneScience/AlphaFold3/":
        raise AssertionError("model repo_id mismatch in conf/alphafold3_paths.yaml")
    if config["dataset"]["repo_id"] != "OneScience/AlphaFold3_dataset":
        raise AssertionError("dataset repo_id mismatch in conf/alphafold3_paths.yaml")

    for rel, (size, digest) in MODEL_FILES.items():
        path = model_dir / rel
        if not path.is_file():
            raise AssertionError(f"missing model file: {path}")
        if path.stat().st_size != size:
            raise AssertionError(f"model size mismatch: {path}")
        if not args.skip_hash and sha256(path) != digest:
            raise AssertionError(f"model sha256 mismatch: {path}")

    for rel in DATASET_REQUIRED:
        path = dataset_root / rel
        if not path.exists():
            raise AssertionError(f"missing dataset dependency: {path}")

    check_json(root / "inputs" / "7r6r_data.json")
    check_json(root / "inputs" / "t1119_search.json")
    print("model_preflight_ok: true")


if __name__ == "__main__":
    main()
