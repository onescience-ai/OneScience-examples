#!/usr/bin/env python3
"""Preflight checks for the standardized SimpleFold ModelScope package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:  # pragma: no cover - environment diagnostic
    print(f"[FAIL] PyYAML is required to parse manifests and configs: {exc}")
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "onescience_run_manifest.yaml",
    "manifest.yaml",
    "configs/train.yaml",
    "configs/data/pdb_sp.yaml",
    "configs/model/architecture/foldingdit_100M.yaml",
    "configs/model/architecture/foldingdit_1.6B.yaml",
    "configs/model/architecture/plddt_module.yaml",
    "inference.py",
    "cli.py",
    "wrapper.py",
    "train.py",
    "train_fsdp.py",
    "assets/7ftv_A.cif",
    "examples/minimal.fasta",
]

REQUIRED_CHECKPOINTS = [
    "simplefold_100M.ckpt",
    "simplefold_360M.ckpt",
    "simplefold_700M.ckpt",
    "simplefold_1.1B.ckpt",
    "simplefold_1.6B.ckpt",
    "simplefold_3B.ckpt",
    "plddt.ckpt",
    "plddt_module_1.6B.ckpt",
    "ccd.pkl",
]


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        fail(f"YAML parse failed for {path.relative_to(ROOT)}: {exc}")
    if not isinstance(data, dict):
        fail(f"YAML root must be a mapping: {path.relative_to(ROOT)}")
    return data


def check_required_files() -> None:
    missing = [item for item in REQUIRED_FILES if not (ROOT / item).exists()]
    if missing:
        fail("Missing required runtime files: " + ", ".join(missing))
    ok("Required runtime files exist")


def check_checkpoints() -> None:
    missing = []
    empty = []
    for name in REQUIRED_CHECKPOINTS:
        path = ROOT / "checkpoints" / name
        if not path.exists():
            missing.append(name)
            continue
        if path.stat().st_size <= 0:
            empty.append(name)
    if missing:
        fail("Missing checkpoints: " + ", ".join(missing))
    if empty:
        fail("Empty checkpoints: " + ", ".join(empty))
    ok("Checkpoint files are present and non-empty")


def check_manifest() -> None:
    manifest = load_yaml(ROOT / "onescience_run_manifest.yaml")
    if manifest.get("resource", {}).get("id") != "OneScience/simplefold/":
        fail("Model resource.id must be exactly OneScience/simplefold/")
    primary = manifest.get("platform_resource", {}).get("primary", {})
    if primary.get("repo_id") != "OneScience/simplefold/":
        fail("platform_resource.primary.repo_id must be exactly OneScience/simplefold/")
    commands = manifest.get("commands", {})
    for scenario in manifest.get("run_matrix", {}).get("scenarios", []):
        for ref in scenario.get("command_refs", []):
            _, stage, name = ref.split(".", 2)
            matches = [cmd for cmd in commands.get(stage, []) if cmd.get("name") == name]
            if len(matches) != 1:
                fail(f"command_ref does not resolve uniquely: {ref}")
    ok("Manifest identity and command_refs are valid")


def check_training_data_policy() -> None:
    cfg = load_yaml(ROOT / "configs/data/pdb_sp.yaml")
    datasets = cfg.get("datasets", [])
    if not datasets:
        fail("configs/data/pdb_sp.yaml must keep the training dataset interface")
    first = datasets[0]
    expected = {
        "tokenized_dir": "./datasets/tokenized",
        "target_dir": "./datasets",
        "manifest_path": "./datasets/manifest.json",
    }
    for key, value in expected.items():
        if first.get(key) != value:
            fail(f"Unexpected training data config {key}: {first.get(key)!r}")
    ok("Training data config interface is preserved; no bundled data is required")


def check_optional_esm() -> None:
    base = Path(os.getenv("ONESCIENCE_MODELS_DIR", "/public/share/sugonhpcapp01/onestore/onemodels"))
    esm = Path(os.getenv("SIMPLEFOLD_ESM2_MODEL_PATH", base / "esm_models" / "esm2_t36_3B_UR50D.pt"))
    regression = Path(f"{esm.with_suffix('')}-contact-regression.pt")
    if esm.exists() and regression.exists():
        ok(f"Local ESM-2 dependency found: {esm}")
    else:
        print("[WARN] Local ESM-2 dependency was not found; inference may download it from the configured hub.")


def main() -> None:
    check_required_files()
    check_checkpoints()
    check_manifest()
    check_training_data_policy()
    check_optional_esm()
    ok("SimpleFold model package preflight passed")


if __name__ == "__main__":
    main()
