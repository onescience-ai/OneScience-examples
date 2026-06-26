#!/usr/bin/env python3
"""Preflight checks for the PINNsformer ModelScope runtime package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REQUIRED_SOURCE_FILES = [
    "README.md",
    "onescience_run_manifest.yaml",
    "manifest.yaml",
    "data.sh",
    "1d_reaction/1d_reaction_pinnsformer.py",
    "1d_reaction/1d_reaction_pinns.py",
    "1d_reaction/1d_reaction_qres.py",
    "1d_reaction/1d_reaction_fls.py",
    "1d_wave/1d_wave_pinnsformer.py",
    "1d_wave/1d_wave_pinnsformer_ntk.py",
    "1d_wave/1d_wave_pinn.py",
    "1d_wave/1d_wave_pinn_ntk.py",
    "convection/convection_pinnsformer.py",
    "convection/convection_pinns.py",
    "convection/convection_qres.py",
    "convection/convection_fls.py",
    "navier_stokes/navier_stoke_pinnsformer.py",
    "navier_stokes/navier_stoke_pinn.py",
    "navier_stokes/navier_stoke_qres.py",
    "navier_stokes/navier_stoke_fls.py",
]

DATA_FILES = {
    "convection/convection.mat": {
        "min_size": 1024,
        "required_keys": {"__header__", "__version__", "__globals__"},
    },
    "navier_stokes/cylinder_nektar_wake.mat": {
        "min_size": 1024,
        "required_keys": {"__header__", "__version__", "__globals__", "U_star", "p_star", "t", "X_star"},
    },
}


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def check_required_files(repo_root: Path) -> None:
    missing = [path for path in REQUIRED_SOURCE_FILES if not (repo_root / path).is_file()]
    if missing:
        fail("missing required source file(s): " + ", ".join(missing))


def check_yaml(repo_root: Path) -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency-specific message
        fail(f"pyyaml import failed: {exc}")

    for rel_path in ["onescience_run_manifest.yaml", "manifest.yaml"]:
        path = repo_root / rel_path
        with path.open("r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
        if not isinstance(doc, dict):
            fail(f"{rel_path} does not parse to a mapping")
        resource = doc.get("resource") or {}
        repo_id = (doc.get("platform_resource") or {}).get("primary", {}).get("repo_id")
        if resource.get("id") != "OneScience/PINNsformer":
            fail(f"{rel_path} resource.id mismatch: {resource.get('id')!r}")
        if repo_id != "OneScience/PINNsformer":
            fail(f"{rel_path} platform_resource.primary.repo_id mismatch: {repo_id!r}")


def check_data_files(repo_root: Path, required: bool) -> None:
    missing = []
    for rel_path, spec in DATA_FILES.items():
        path = repo_root / rel_path
        if not path.is_file():
            missing.append(rel_path)
            continue
        if path.stat().st_size < int(spec["min_size"]):
            fail(f"data file is unexpectedly small: {rel_path}")
    if missing and required:
        fail("missing data file(s): " + ", ".join(missing))
    if missing:
        print("[WARN] data file(s) not found; run bash data.sh before data-dependent examples: " + ", ".join(missing))


def check_data_schema(repo_root: Path) -> None:
    try:
        import scipy.io
    except Exception as exc:  # pragma: no cover - dependency-specific message
        fail(f"scipy import failed: {exc}")

    for rel_path, spec in DATA_FILES.items():
        path = repo_root / rel_path
        if not path.is_file():
            fail(f"missing data file for schema check: {rel_path}")
        mat = scipy.io.loadmat(path)
        required_keys = set(spec["required_keys"])
        missing_keys = sorted(required_keys.difference(mat.keys()))
        if missing_keys:
            fail(f"{rel_path} missing key(s): {', '.join(missing_keys)}")
        user_keys = sorted(key for key in mat if not key.startswith("__"))
        print(f"[OK] {rel_path} keys: {', '.join(user_keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight PINNsformer runtime package.")
    parser.add_argument("--repo-root", default=".", help="Path to the PINNsformer package root.")
    parser.add_argument("--require-data", action="store_true", help="Fail when data files are missing.")
    parser.add_argument("--check-data-schema", action="store_true", help="Load .mat files and validate key variables.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        fail(f"repo root does not exist: {repo_root}")

    check_required_files(repo_root)
    check_yaml(repo_root)
    check_data_files(repo_root, required=args.require_data or args.check_data_schema)
    if args.check_data_schema:
        check_data_schema(repo_root)

    print("[OK] model preflight completed")


if __name__ == "__main__":
    main()
