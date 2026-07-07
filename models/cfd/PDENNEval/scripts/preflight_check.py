#!/usr/bin/env python3
"""Preflight checks for the PDENNEval standard runtime package."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_ROOT = REPO_ROOT / "conf"
DATA_ENV = "ONESCIENCE_PDENNEVAL_DATA_DIR"


EXPECTED_CONFIGS = {
    "fno_2d_darcy.yaml": ("fno_config", "2D_DarcyFlow_beta0.1_Train.hdf5", "2D_Darcy_Flow"),
    "fno_1d_burgers.yaml": ("fno_config", "1D_Burgers_Sols_Nu0.001.hdf5", "1D_Burgers"),
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        fail(f"config must be a YAML mapping: {path}")
    return data


def resolve_data_root(expr: str) -> Path:
    if expr != f"${{{DATA_ENV}}}":
        fail(f"datapipe.source.data_dir must be ${{{DATA_ENV}}}, got {expr!r}")
    value = os.environ.get(DATA_ENV)
    if not value:
        fail(f"{DATA_ENV} is not set")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        fail(f"{DATA_ENV} does not point to a directory: {root}")
    return root


def validate_hdf5(path: Path, pde_name: str) -> None:
    if not path.is_file():
        fail(f"missing HDF5 data file: {path}")
    with h5py.File(path, "r") as handle:
        if "tensor" not in handle:
            fail(f"{path.name} must contain dataset 'tensor'")
        tensor = handle["tensor"]
        if pde_name == "2D_Darcy_Flow":
            if tensor.ndim != 4 or "nu" not in handle:
                fail(f"{path.name} must be 2D Darcy Flow data with tensor ndim=4 and nu dataset")
        elif pde_name == "1D_Burgers":
            if tensor.ndim != 3 or "x-coordinate" not in handle or "t-coordinate" not in handle:
                fail(f"{path.name} must be 1D Burgers data with tensor/x-coordinate/t-coordinate")
        else:
            fail(f"unsupported standardized preflight pde_name: {pde_name}")
    ok(f"{path.name} matches {pde_name} schema")


def validate_config(config_name: str, root_key: str, expected_file: str, expected_pde: str) -> Path:
    path = CONF_ROOT / config_name
    data = load_yaml(path)
    if root_key not in data:
        fail(f"{config_name} missing root key {root_key!r}")
    cfg = data[root_key]
    source = cfg.get("datapipe", {}).get("source", {})
    data_cfg = cfg.get("datapipe", {}).get("data", {})
    training = cfg.get("training", {})
    data_root = resolve_data_root(source.get("data_dir"))
    if source.get("file_name") != expected_file:
        fail(f"{config_name} file_name expected {expected_file}, got {source.get('file_name')!r}")
    if data_cfg.get("pde_name") != expected_pde:
        fail(f"{config_name} pde_name expected {expected_pde}, got {data_cfg.get('pde_name')!r}")
    output_dir = Path(training.get("output_dir", "./checkpoint/"))
    output_path = output_dir if output_dir.is_absolute() else (REPO_ROOT / output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    probe = output_path / ".preflight_write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    validate_hdf5(data_root / expected_file, expected_pde)
    ok(f"{config_name} is valid for the standardized package")
    return data_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(EXPECTED_CONFIGS), default="fno_2d_darcy.yaml")
    parser.add_argument("--all", action="store_true", help="validate every standardized config")
    args = parser.parse_args()

    items = EXPECTED_CONFIGS.items() if args.all else [(args.config, EXPECTED_CONFIGS[args.config])]
    seen_roots = set()
    for config_name, (root_key, expected_file, expected_pde) in items:
        data_root = validate_config(config_name, root_key, expected_file, expected_pde)
        seen_roots.add(str(data_root))
    warn("training and evaluation require the OneScience CFD runtime, PyTorch, h5py and enough memory for multi-GB HDF5 files")
    ok(f"model preflight completed; validated data roots: {', '.join(sorted(seen_roots))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
