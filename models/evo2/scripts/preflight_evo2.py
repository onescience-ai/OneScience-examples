#!/usr/bin/env python3
import argparse
import hashlib
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required for Evo2 preflight: {exc}")

REQUIRED_MODEL_FILES = [
    "context/model.yaml",
    "context/io.json",
    "weights/.metadata",
    "weights/__0_0.distcp",
    "weights/__0_1.distcp",
    "weights/common.pt",
    "weights/metadata.json",
]

REQUIRED_DATA_PREFIXES = [
    "preprocessed_data/chr20_21_22_uint8_distinct_byte-level_train",
    "preprocessed_data/chr20_21_22_uint8_distinct_byte-level_val",
    "preprocessed_data/chr20_21_22_uint8_distinct_byte-level_test",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_sha_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(None, 1)
        result[filename.strip()] = digest
    return result


def check_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for the OneScience Evo2 ModelScope package.")
    parser.add_argument("--package-root", default=".", help="Model package root. Default: current directory.")
    parser.add_argument("--data-root", default="data/evo2_dataset/data_mini/genome_data", help="Genome data directory from OneScience/evo2_dataset.")
    parser.add_argument("--skip-sha256", action="store_true", help="Skip SHA256 checks for large checkpoint files.")
    args = parser.parse_args()

    root = Path(args.package_root).resolve()
    data_root = (root / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    ckpt_root = root / "checkpoints" / "evo2_nemo_7b"
    errors: list[str] = []

    for rel in ["manifest.yaml", "README.md", "config/genome_data_config.yaml", "config/genome_preprocess_config.yaml", "config/opengenome2.yml"]:
        if not (root / rel).exists():
            errors.append(f"missing required package file: {rel}")

    try:
        genome_cfg = check_yaml(root / "config" / "genome_data_config.yaml")
        if len(genome_cfg) != 3:
            errors.append("config/genome_data_config.yaml must contain train, validation and test entries.")
    except Exception as exc:
        errors.append(f"cannot parse config/genome_data_config.yaml: {exc}")

    try:
        adapted_cfg = check_yaml(root / "config" / "opengenome2.yml")
        for key in ["train-data-paths", "valid-data-paths", "test-data-paths"]:
            values = adapted_cfg.get(key, [])
            if len(values) != 1 or "chr20_21_22_uint8_distinct_byte-level" not in values[0]:
                errors.append(f"config/opengenome2.yml {key} is not adapted to chr20/21/22 mini data.")
    except Exception as exc:
        errors.append(f"cannot parse config/opengenome2.yml: {exc}")

    for rel in REQUIRED_MODEL_FILES:
        if not (ckpt_root / rel).is_file():
            errors.append(f"missing model checkpoint file: checkpoints/evo2_nemo_7b/{rel}")

    for prefix in REQUIRED_DATA_PREFIXES:
        for suffix in [".bin", ".idx"]:
            if not (data_root / f"{prefix}{suffix}").is_file():
                errors.append(f"missing dataset file: {data_root / f'{prefix}{suffix}'}")

    if not args.skip_sha256:
        sha_manifest = root / "metadata" / "evo2_nemo_7b.sha256"
        if sha_manifest.is_file():
            expected = read_sha_manifest(sha_manifest)
            for rel, digest in expected.items():
                actual_path = ckpt_root / rel
                if actual_path.is_file() and sha256(actual_path) != digest:
                    errors.append(f"sha256 mismatch: checkpoints/evo2_nemo_7b/{rel}")
        else:
            errors.append("missing metadata/evo2_nemo_7b.sha256")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1

    print("[OK] Evo2 model package, checkpoint files, adapted configs and dataset paths passed preflight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
