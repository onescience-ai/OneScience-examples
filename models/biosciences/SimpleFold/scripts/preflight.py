#!/usr/bin/env python3
"""Preflight checks for the packaged SimpleFold ModelScope layout."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    "README.md",
    "configuration.json",
    "requirements_dcu.txt",
    "env_install.sh",
    "config/train.yaml",
    "config/base_train.yaml",
    "config/base_eval.yaml",
    "config/model/architecture/foldingdit_100M.yaml",
    "config/model/architecture/foldingdit_1.6B.yaml",
    "config/model/architecture/plddt_module.yaml",
    "scripts/run_inference.py",
    "scripts/inference.py",
    "scripts/train.py",
    "scripts/train_fsdp.py",
    "scripts/evaluate.py",
    "scripts/process_data.py",
    "scripts/tokenize_data.py",
    "examples/minimal.fasta",
    "modules/models/simplefold/flow.py",
    "modules/datapipes/simplefold/processor/protein_processor.py",
    "modules/utils/simplefold/esm_utils.py",
    "modules/datapipes/boltz_data_pipeline/types.py",
    "modules/models/esm/pretrained.py",
]

REQUIRED_WEIGHTS = [
    "simplefold_100M.ckpt",
    "simplefold_360M.ckpt",
    "simplefold_700M.ckpt",
    "simplefold_1.1B.ckpt",
    "simplefold_1.6B.ckpt",
    "simplefold_3B.ckpt",
    "plddt.ckpt",
    "plddt_module_1.6B.ckpt",
    "ccd.pkl",
    "boltz1_conf.ckpt",
]

OPTIONAL_ESM_WEIGHTS = [
    "esm_models/esm2_t36_3B_UR50D.pt",
    "esm_models/esm2_t36_3B_UR50D-contact-regression.pt",
]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def check_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        fail("Missing required files: " + ", ".join(missing))
    ok("Required package files are present")


def check_weights(strict: bool) -> None:
    missing = []
    placeholders = []
    for name in REQUIRED_WEIGHTS:
        path = ROOT / "weight" / name
        if not path.exists():
            missing.append(name)
        elif path.stat().st_size < 1024:
            placeholders.append(name)
    if missing:
        fail("Missing weight files: " + ", ".join(missing))
    if placeholders:
        message = (
            "These files look like link/placeholder files and must be replaced "
            "before real inference/training: " + ", ".join(placeholders)
        )
        if strict:
            fail(message)
        warn(message)
    else:
        ok("Required SimpleFold weights look like real files")

    missing_esm = [name for name in OPTIONAL_ESM_WEIGHTS if not (ROOT / "weight" / name).exists()]
    if missing_esm:
        warn(
            "ESM-2 3B local weights are not present. Put them under weight/esm_models "
            "before running inference: " + ", ".join(missing_esm)
        )
    else:
        ok("Local ESM-2 3B files are present")


def check_imports(strict: bool) -> None:
    modules = [
        "modules.models.simplefold.flow",
        "modules.models.simplefold.torch.architecture",
        "modules.datapipes.simplefold.processor.protein_processor",
        "modules.utils.simplefold.esm_utils",
        "modules.datapipes.boltz_data_pipeline.types",
        "modules.models.esm.pretrained",
    ]
    missing_dependencies = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_dependencies.append(f"{module_name}: {exc.name}")
    if missing_dependencies:
        message = (
            "Some imports need external Python dependencies. Install requirements_dcu.txt "
            "and rerun with --strict-imports for a full check: "
            + "; ".join(missing_dependencies)
        )
        if strict:
            fail(message)
        warn(message)
    else:
        ok("Bundled Python modules import successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check packaged SimpleFold runtime files.")
    parser.add_argument("--strict-weights", action="store_true")
    parser.add_argument("--strict-imports", action="store_true")
    args = parser.parse_args()

    check_files()
    check_weights(strict=args.strict_weights)
    check_imports(strict=args.strict_imports)
    ok("SimpleFold package preflight finished")


if __name__ == "__main__":
    main()
