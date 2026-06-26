#!/usr/bin/env python3
"""Preflight checks for the Transolver Airfoil Design runtime package."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "transolver_airfrans.yaml"
REQUIRED_MANIFEST_KEYS = ("full_train", "full_test")
REQUIRED_SUFFIXES = ("internal.vtu", "freestream.vtp", "aerofoil.vtp")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def load_text(path: Path) -> str:
    if not path.exists():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def extract_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*['\"]?([^'\"\n#]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


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


def check_config() -> tuple[str, Path, Path]:
    text = load_text(CONFIG_PATH)
    model_name = extract_scalar(text, "name")
    if model_name != "Transolver":
        fail(f"unexpected model.name in {CONFIG_PATH}: {model_name!r}")
    if "Transolver:" not in text or "Transolver_plus:" not in text:
        fail("config is missing Transolver or Transolver_plus specific_params")

    data_dir_expr = extract_nested_value(text, "source", "data_dir")
    stats_dir_expr = extract_nested_value(text, "source", "stats_dir")
    if data_dir_expr != "${ONESCIENCE_AIRFRANS_DATA_DIR}":
        fail(f"datapipe.source.data_dir must be ${{ONESCIENCE_AIRFRANS_DATA_DIR}}, got {data_dir_expr!r}")
    if not stats_dir_expr:
        fail("datapipe.source.stats_dir is missing")

    data_env = os.environ.get("ONESCIENCE_AIRFRANS_DATA_DIR")
    if not data_env:
        fail("ONESCIENCE_AIRFRANS_DATA_DIR is not set")
    data_dir = Path(data_env).expanduser().resolve()
    stats_dir = (REPO_ROOT / stats_dir_expr).resolve() if not Path(stats_dir_expr).is_absolute() else Path(stats_dir_expr)
    ok("config file and model selection are valid")
    return data_dir_expr, data_dir, stats_dir


def check_dataset(data_dir: Path) -> None:
    if not data_dir.exists():
        fail(f"dataset directory does not exist: {data_dir}")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        fail(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest or not isinstance(manifest[key], list) or not manifest[key]:
            fail(f"manifest key is missing or empty: {key}")

    referenced = []
    for key in REQUIRED_MANIFEST_KEYS:
        referenced.extend(manifest[key])
    missing = []
    for sim_name in referenced:
        sim_dir = data_dir / sim_name
        if not sim_dir.is_dir():
            missing.append(str(sim_dir))
            continue
        for suffix in REQUIRED_SUFFIXES:
            file_path = sim_dir / f"{sim_name}_{suffix}"
            if not file_path.is_file():
                missing.append(str(file_path))
    if missing:
        fail("missing required dataset files, first entries: " + "; ".join(missing[:8]))

    actual_cases = [p for p in data_dir.iterdir() if p.is_dir()]
    if len(set(referenced)) > len(actual_cases):
        fail("manifest references more cases than the dataset directory contains")
    ok(f"dataset manifest and required files are present: {len(actual_cases)} cases")


def check_stats(stats_dir: Path) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    probe = stats_dir / ".preflight_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        fail(f"stats_dir is not writable: {stats_dir}: {exc}")
    required_stats = ["mean_in.npy", "std_in.npy", "mean_out.npy", "std_out.npy"]
    missing_stats = [name for name in required_stats if not (stats_dir / name).exists()]
    if missing_stats:
        warn("normalization stats are not present yet; training mode will compute them: " + ", ".join(missing_stats))
    ok(f"stats_dir is writable: {stats_dir}")


def check_checkpoint() -> None:
    ckpt_dir = REPO_ROOT / "checkpoints" / "transolver_airfrans"
    ckpt_file = ckpt_dir / "Transolver.pth"
    if ckpt_file.exists():
        ok(f"inference checkpoint exists: {ckpt_file}")
    else:
        warn(f"inference needs a trained checkpoint at {ckpt_file}; train.py can create it")


def main() -> int:
    _, data_dir, stats_dir = check_config()
    check_dataset(data_dir)
    check_stats(stats_dir)
    check_checkpoint()
    ok("model preflight completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
