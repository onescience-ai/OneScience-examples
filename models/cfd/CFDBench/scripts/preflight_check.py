#!/usr/bin/env python3
"""Preflight checks for the CFDBench standard runtime package."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "cfdbench.yaml"
PROBLEMS = {
    "tube": {"bc_key": "vel_in", "param_keys": ["vel_in", "density", "viscosity", "height", "width"]},
    "cavity": {"bc_key": "vel_top", "param_keys": ["vel_top", "density", "viscosity", "height", "width"]},
    "cylinder": {"bc_key": "vel_in", "param_keys": ["vel_in", "density", "viscosity", "height", "width", "x_min", "x_max", "y_min", "y_max", "radius"]},
    "dam": {"bc_key": "velocity", "param_keys": ["velocity", "density", "viscosity", "height", "width"]},
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


def resolve_data_root(expr: str) -> Path:
    if expr != "${ONESCIENCE_CFDBENCH_DATA_DIR}":
        fail(f"datapipe.source.data_dir must be ${{ONESCIENCE_CFDBENCH_DATA_DIR}}, got {expr!r}")
    value = os.environ.get("ONESCIENCE_CFDBENCH_DATA_DIR")
    if not value:
        fail("ONESCIENCE_CFDBENCH_DATA_DIR is not set")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        fail(f"CFDBench data directory does not exist: {root}")
    return root


def parse_config() -> tuple[Path, str, str, str, Path]:
    text = load_config_text()
    data_dir_expr = extract_nested_value(text, "source", "data_dir")
    data_name = extract_nested_value(text, "source", "data_name")
    task_type = extract_scalar(text, "task_type")
    model_name = extract_scalar(text, "name")
    output_dir = extract_nested_value(text, "training", "output_dir") or "./result"

    if data_name != "tube_prop_bc_geo":
        fail(f"standard package expects data_name=tube_prop_bc_geo, got {data_name!r}")
    if task_type != "auto":
        fail(f"standard package expects task_type=auto, got {task_type!r}")
    if model_name != "fno":
        fail(f"standard package expects model.name=fno, got {model_name!r}")
    data_root = resolve_data_root(data_dir_expr or "")
    output_path = (REPO_ROOT / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)
    ok("config file is valid for the standardized CFDBench package")
    return data_root, data_name, task_type, model_name, output_path


def check_case(case_dir: Path, problem: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    for name in ("case.json", "u.npy", "v.npy"):
        if not (case_dir / name).is_file():
            fail(f"missing required file: {case_dir / name}")
    try:
        params = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid case.json: {case_dir / 'case.json'}: {exc}")
    for key in PROBLEMS[problem]["param_keys"]:
        if key not in params:
            fail(f"case.json missing key {key!r}: {case_dir / 'case.json'}")
    u = np.load(case_dir / "u.npy", mmap_mode="r")
    v = np.load(case_dir / "v.npy", mmap_mode="r")
    if u.shape != v.shape:
        fail(f"u/v shape mismatch in {case_dir}: {u.shape} vs {v.shape}")
    if u.ndim != 3:
        fail(f"expected 3D time/grid array in {case_dir}, got shape {u.shape}")
    if not np.issubdtype(u.dtype, np.number) or not np.issubdtype(v.dtype, np.number):
        fail(f"u/v dtype must be numeric in {case_dir}: {u.dtype}, {v.dtype}")
    probe_u = np.asarray(u[0])
    probe_v = np.asarray(v[0])
    if not np.isfinite(probe_u).all() or not np.isfinite(probe_v).all():
        fail(f"first frame contains non-finite values in {case_dir}")
    return tuple(u.shape), tuple(v.shape)


def check_dataset(data_root: Path, data_name: str, sample_cases: int) -> None:
    problem = data_name.split("_")[0]
    subset = data_name[len(problem) + 1 :]
    if problem not in PROBLEMS:
        fail(f"unknown problem in data_name: {problem}")
    selected_dirs = []
    for token in ("prop", "bc", "geo"):
        if token in subset:
            part_dir = data_root / problem / token
            if not part_dir.is_dir():
                fail(f"missing subset directory: {part_dir}")
            cases = sorted(part_dir.glob("case*"), key=lambda p: int(p.name[4:]))
            if not cases:
                fail(f"no cases found in subset directory: {part_dir}")
            selected_dirs.extend(cases)
    if not selected_dirs:
        fail(f"data_name selected no cases: {data_name}")
    for case_dir in selected_dirs[:sample_cases]:
        check_case(case_dir, problem)
    ok(f"dataset subset {data_name} is present: {len(selected_dirs)} cases; sampled {min(sample_cases, len(selected_dirs))}")


def check_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".preflight_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        fail(f"output_dir is not writable: {output_dir}: {exc}")
    ok(f"output_dir is writable: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-cases", type=int, default=3)
    args = parser.parse_args()
    if args.sample_cases < 1:
        fail("--sample-cases must be positive")
    data_root, data_name, _, _, output_dir = parse_config()
    check_dataset(data_root, data_name, args.sample_cases)
    check_output_dir(output_dir)
    warn("training/inference require the OneScience CFD runtime with torch, torch_geometric-compatible dependencies, and sufficient memory for CFDBench")
    ok("model preflight completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
