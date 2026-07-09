#!/usr/bin/env python3
"""Static and file-layout preflight for the standalone Protenix package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path


MODEL_SHA256 = "9ea20b0aba42f2256711da1d0cd081510a4b291e64375bff6b70ced70b87a5f1"
MODEL_SIZE = 1474265486

REQUIRED_PACKAGE_FILES = [
    "README.md",
    "configuration.json",
    "configs/inference_config.yaml",
    "weight/model_v0.5.0.pt",
    "examples/7r6r.json",
    "examples/7r6r/msa/1/pairing.a3m",
    "examples/7r6r/msa/1/non_pairing.a3m",
    "models/protenix/protenix.py",
    "models/openfold/primitives.py",
    "scripts/runner/inference_unified.py",
    "scripts/run_inference.py",
    "scripts/train.py",
    "scripts/finetune.py",
]

REQUIRED_DATA_FILES = [
    "components.v20240608.cif",
    "components.v20240608.cif.rdkit_mol.pkl",
    "seq_to_pdb_index.json",
    "indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz",
    "indices/recentPDB_low_homology_maxtoken1536.csv",
    "indices/recentPDB_low_homology_maxtoken1024_sample384_pdb_id.txt",
    "indices/posebusters_indices_mainchain_interface.csv",
    "mmcif_msa",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_to_path(model_root: Path, module: str) -> Path | None:
    if not module.startswith(("configs.", "models.", "scripts.")):
        return None
    rel = Path(*module.split("."))
    package_init = model_root / rel / "__init__.py"
    module_file = model_root / rel.with_suffix(".py")
    if package_init.exists():
        return package_init
    if module_file.exists():
        return module_file
    return module_file


def check_local_imports(model_root: Path) -> list[str]:
    missing: list[str] = []
    scan_roots = ["configs", "models", "scripts"]
    py_files = []
    for scan_root in scan_roots:
        py_files.extend((model_root / scan_root).rglob("*.py"))
    for py_file in sorted(py_files):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if module == "modules" or module.startswith("modules."):
                        missing.append(f"{py_file.relative_to(model_root)} imports removed package {module}")
                        continue
                    if module == "config" or module.startswith("config."):
                        missing.append(f"{py_file.relative_to(model_root)} imports renamed package {module}")
                        continue
                    target = module_to_path(model_root, module)
                    if target is not None and not target.exists():
                        missing.append(f"{py_file.relative_to(model_root)} imports missing {module}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if node.level == 0:
                    if module == "modules" or module.startswith("modules."):
                        missing.append(f"{py_file.relative_to(model_root)} imports removed package {module}")
                        continue
                    if module == "config" or module.startswith("config."):
                        missing.append(f"{py_file.relative_to(model_root)} imports renamed package {module}")
                        continue
                target = module_to_path(model_root, module)
                if target is not None and not target.exists():
                    missing.append(f"{py_file.relative_to(model_root)} imports missing {module}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Protenix standalone package preflight")
    parser.add_argument("--model-root", default=".", help="Package root")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT_DIR", "../bio_protenix_dataset"),
        help="Prepared Protenix dataset root",
    )
    parser.add_argument("--strict-weights", action="store_true", help="Validate weight size and LFS pointer state")
    parser.add_argument("--full-checksum", action="store_true", help="Compute SHA256 for the 1.4GB model file")
    parser.add_argument("--strict-imports", action="store_true", help="Statically verify local imports and removed modules.* imports")
    parser.add_argument("--strict-data", action="store_true", help="Require the external dataset files")
    args = parser.parse_args()

    model_root = Path(args.model_root).resolve()
    data_root = Path(args.data_root).resolve()
    errors: list[str] = []

    for rel in REQUIRED_PACKAGE_FILES:
        path = model_root / rel
        if not path.exists():
            errors.append(f"Missing package file: {path}")

    config = model_root / "configs/inference_config.yaml"
    if config.exists():
        text = config.read_text(encoding="utf-8")
        required_fragments = [
            'input_json_path: "./examples/7r6r.json"',
            'load_checkpoint_path: "./weight/model_v0.5.0.pt"',
            'ccd_components_file: "${DATA_ROOT_DIR}/components.v20240608.cif"',
            'ccd_components_rdkit_mol_file: "${DATA_ROOT_DIR}/components.v20240608.cif.rdkit_mol.pkl"',
            'pdb_mmseqs_dir: "${DATA_ROOT_DIR}/mmcif_msa"',
            "${oc.env:PWD}/examples/7r6r/msa/1",
        ]
        for fragment in required_fragments:
            if fragment not in text:
                errors.append(f"Config missing fragment: {fragment}")

    checkpoint = model_root / "weight/model_v0.5.0.pt"
    if args.strict_weights and checkpoint.exists():
        size = checkpoint.stat().st_size
        if size != MODEL_SIZE:
            errors.append(f"Weight size mismatch: {checkpoint} size={size} expected={MODEL_SIZE}")
        with checkpoint.open("rb") as f:
            prefix = f.read(64)
        if prefix.startswith(b"version https://git-lfs.github.com/spec"):
            errors.append(f"Weight is a Git LFS pointer, not the real checkpoint: {checkpoint}")
        if args.full_checksum:
            digest = sha256(checkpoint)
            if digest != MODEL_SHA256:
                errors.append(f"Weight SHA256 mismatch: {checkpoint} sha256={digest}")

    if args.strict_data:
        for rel in REQUIRED_DATA_FILES:
            path = data_root / rel
            if not path.exists():
                errors.append(f"Missing dataset file or directory: {path}")

    if args.strict_imports:
        errors.extend(check_local_imports(model_root))

    residue_patterns = [
        "from modules.",
        "import modules.",
        "modules.models.",
        "modules.runner.",
        "/public/share/sugonhpcapp01/onestore/" + "onemodels",
        "/public/home/liuyx19/" + "one" + "science",
        "checkpoints/" + "model_v0.5.0.pt",
        "infer_" + "datasets/" + "7r6r",
        "url" + "retrieve",
    ]
    for path in sorted(model_root.rglob("*")):
        if not path.is_file() or "weight" in path.parts:
            continue
        if "modules" in path.relative_to(model_root).parts:
            continue
        if path.relative_to(model_root).as_posix() == "scripts/preflight.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in residue_patterns:
            if pattern in text:
                errors.append(f"Residual pattern {pattern!r} in {path.relative_to(model_root)}")

    print(f"model_root={model_root}")
    print(f"data_root={data_root}")
    if errors:
        print("Preflight failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
