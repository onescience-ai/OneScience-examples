#!/usr/bin/env python3
"""Preflight checks for the BENO standard runtime package."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "beno.yaml"
REQUIRED_KINDS = ("RHS", "SOL", "BC")
EXPECTED_SHAPES = {
    "RHS": (1000, 1024, 4),
    "SOL": (1000, 1024, 1),
    "BC": (1000, 128, 4),
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def load_config_text() -> str:
    if not CONFIG_PATH.exists():
        fail(f"missing config file: {CONFIG_PATH}")
    return CONFIG_PATH.read_text(encoding="utf-8")


def extract_nested_value(text: str, section: str, key: str) -> str | None:
    in_section = False
    section_indent = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if re.match(rf"^\s*{re.escape(section)}:\s*$", line):
            in_section = True
            section_indent = indent
            continue
        if in_section and section_indent is not None and indent <= section_indent:
            in_section = False
        if in_section and re.match(rf"^\s*{re.escape(key)}:\s*", line):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def extract_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*['\"]?([^'\"\n#]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def resolve_data_dir(expr: str) -> Path:
    if expr != "${ONESCIENCE_BENO_DATA_DIR}/Dirichlet/":
        fail(f"datapipe.source.data_dir must be ${{ONESCIENCE_BENO_DATA_DIR}}/Dirichlet/, got {expr!r}")
    dataset_root = os.environ.get("ONESCIENCE_BENO_DATA_DIR")
    if not dataset_root:
        fail("ONESCIENCE_BENO_DATA_DIR is not set")
    data_dir = Path(dataset_root).expanduser().resolve() / "Dirichlet"
    if not data_dir.is_dir():
        fail(f"Dirichlet dataset directory does not exist: {data_dir}")
    return data_dir


def check_config() -> tuple[Path, str, int, int, int, Path]:
    text = load_config_text()
    data_dir_expr = extract_nested_value(text, "source", "data_dir")
    cache_dir_expr = extract_nested_value(text, "source", "cache_dir")
    file_prefix = extract_nested_value(text, "source", "file_prefix")
    ntrain = extract_scalar(text, "ntrain")
    ntest = extract_scalar(text, "ntest")
    resolution = extract_scalar(text, "resolution")
    model_name = extract_scalar(text, "name")

    if model_name != "BENO_Elliptic":
        fail(f"unexpected datapipe.data.name: {model_name!r}")
    if file_prefix != "N32_4c":
        fail(f"this runtime package is configured for file_prefix N32_4c, got {file_prefix!r}")
    if (ntrain, ntest, resolution) != ("900", "100", "32"):
        fail(f"unexpected split/resolution: ntrain={ntrain}, ntest={ntest}, resolution={resolution}")
    if not cache_dir_expr:
        fail("datapipe.source.cache_dir is missing")

    data_dir = resolve_data_dir(data_dir_expr or "")
    cache_dir = (REPO_ROOT / cache_dir_expr).resolve() if not Path(cache_dir_expr).is_absolute() else Path(cache_dir_expr)
    ok("config file is valid for the standardized BENO package")
    return data_dir, file_prefix, int(ntrain), int(ntest), int(resolution), cache_dir


def check_npy_files(data_dir: Path, file_prefix: str, ntrain: int, ntest: int, resolution: int) -> None:
    total_required = ntrain + ntest
    expected_nodes = resolution * resolution
    for kind in REQUIRED_KINDS:
        path = data_dir / f"{kind}_{file_prefix}_all.npy"
        if not path.is_file():
            fail(f"missing required data file: {path}")
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != np.float64:
            fail(f"{path.name} dtype must be float64, got {arr.dtype}")
        if tuple(arr.shape) != EXPECTED_SHAPES[kind]:
            fail(f"{path.name} shape mismatch: expected {EXPECTED_SHAPES[kind]}, got {tuple(arr.shape)}")
        if arr.shape[0] < total_required:
            fail(f"{path.name} has fewer samples than ntrain+ntest={total_required}")
        if kind in ("RHS", "SOL") and arr.shape[1] != expected_nodes:
            fail(f"{path.name} node count does not match resolution^2={expected_nodes}")
    ok(f"required NPY files are present and match schema for {file_prefix}")


def check_optional_checkpoint() -> None:
    ckpt_dir = REPO_ROOT / "model"
    checkpoints = sorted(ckpt_dir.glob("model_epoch_*.pt")) if ckpt_dir.exists() else []
    if checkpoints:
        ok(f"checkpoint exists for inference: {checkpoints[-1]}")
    else:
        warn("inference will use random weights until train.py creates model/model_epoch_*.pt")


def check_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = cache_dir / ".preflight_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        fail(f"cache_dir is not writable: {cache_dir}: {exc}")
    ok(f"cache_dir is writable: {cache_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    data_dir, file_prefix, ntrain, ntest, resolution, cache_dir = check_config()
    check_npy_files(data_dir, file_prefix, ntrain, ntest, resolution)
    check_cache_dir(cache_dir)
    if not args.skip_checkpoint:
        check_optional_checkpoint()
    ok("model preflight completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
