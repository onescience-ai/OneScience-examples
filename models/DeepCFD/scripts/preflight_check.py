#!/usr/bin/env python3
"""Preflight checks for the DeepCFD standard runtime package."""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "deepcfd.yaml"
EXPECTED_SHAPE = (981, 3, 172, 79)
EXPECTED_DTYPE = np.float32


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
    if expr != "${ONESCIENCE_DEEPCFD_DATA_DIR}":
        fail(f"datapipe.source.data_dir must be ${{ONESCIENCE_DEEPCFD_DATA_DIR}}, got {expr!r}")
    dataset_root = os.environ.get("ONESCIENCE_DEEPCFD_DATA_DIR")
    if not dataset_root:
        fail("ONESCIENCE_DEEPCFD_DATA_DIR is not set")
    data_dir = Path(dataset_root).expanduser().resolve()
    if not data_dir.is_dir():
        fail(f"DeepCFD dataset directory does not exist: {data_dir}")
    return data_dir


def check_config() -> tuple[Path, str, str, float, int, int]:
    text = load_config_text()
    data_dir_expr = extract_nested_value(text, "source", "data_dir")
    data_x_name = extract_nested_value(text, "source", "data_x_name")
    data_y_name = extract_nested_value(text, "source", "data_y_name")
    split_ratio = extract_scalar(text, "split_ratio")
    batch_size = extract_scalar(text, "batch_size")
    num_workers = extract_scalar(text, "num_workers")
    model_name = extract_scalar(text, "name")
    in_channels = extract_scalar(text, "in_channels")
    out_channels = extract_scalar(text, "out_channels")

    if (data_x_name, data_y_name) != ("dataX.pkl", "dataY.pkl"):
        fail(f"unexpected data file names: data_x_name={data_x_name}, data_y_name={data_y_name}")
    if model_name != "UNetEx":
        fail(f"unexpected model.name: {model_name!r}")
    if (in_channels, out_channels) != ("3", "3"):
        fail(f"unexpected channel configuration: in={in_channels}, out={out_channels}")
    if split_ratio != "0.7":
        fail(f"unexpected split_ratio: {split_ratio}")
    if not batch_size or int(batch_size) <= 0:
        fail(f"invalid dataloader.batch_size: {batch_size}")
    if num_workers is None or int(num_workers) < 0:
        fail(f"invalid dataloader.num_workers: {num_workers}")

    data_dir = resolve_data_dir(data_dir_expr or "")
    ok("config file is valid for the standardized DeepCFD package")
    return data_dir, data_x_name, data_y_name, float(split_ratio), int(batch_size), int(num_workers)


def load_pickle_array(path: Path, mmap_sample: bool = False) -> np.ndarray:
    if not path.is_file():
        fail(f"missing required data file: {path}")
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    arr = np.asarray(obj)
    if arr.shape != EXPECTED_SHAPE:
        fail(f"{path.name} shape mismatch: expected {EXPECTED_SHAPE}, got {arr.shape}")
    if arr.dtype != EXPECTED_DTYPE:
        fail(f"{path.name} dtype mismatch: expected {EXPECTED_DTYPE}, got {arr.dtype}")
    if not mmap_sample and not np.isfinite(arr[0]).all():
        fail(f"{path.name} first sample contains non-finite values")
    return arr


def check_data_files(data_dir: Path, data_x_name: str, data_y_name: str, split_ratio: float) -> None:
    x = load_pickle_array(data_dir / data_x_name)
    y = load_pickle_array(data_dir / data_y_name)
    if x.shape != y.shape:
        fail(f"dataX/dataY shape mismatch: {x.shape} vs {y.shape}")
    split_idx = int(x.shape[0] * split_ratio)
    if split_idx != 686 or x.shape[0] - split_idx != 295:
        fail(f"unexpected split sizes for 981 samples and split_ratio 0.7: train={split_idx}, test={x.shape[0] - split_idx}")
    channel_weights = np.sqrt(np.mean(np.transpose(y, (0, 2, 3, 1)).reshape((-1, 3)) ** 2, axis=0))
    if channel_weights.shape != (3,) or not np.isfinite(channel_weights).all() or np.any(channel_weights <= 0):
        fail(f"invalid loss channel weights computed from dataY.pkl: {channel_weights}")
    ok("required PKL files are present, readable and match DeepCFD schema")
    ok(f"split is valid: train={split_idx}, test={x.shape[0] - split_idx}")


def check_optional_checkpoint() -> None:
    checkpoint = REPO_ROOT / "result" / "deepcfd" / "best_model.pt"
    if checkpoint.exists():
        ok(f"checkpoint exists for inference: {checkpoint}")
    else:
        warn("inference.py requires result/deepcfd/best_model.pt; run train.py first or provide a compatible checkpoint")


def check_output_dir() -> None:
    output_dir = REPO_ROOT / "result" / "deepcfd"
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".preflight_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        fail(f"training output_dir is not writable: {output_dir}: {exc}")
    ok(f"training output_dir is writable: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()
    data_dir, data_x_name, data_y_name, split_ratio, _batch_size, _num_workers = check_config()
    check_data_files(data_dir, data_x_name, data_y_name, split_ratio)
    check_output_dir()
    if not args.skip_checkpoint:
        check_optional_checkpoint()
    ok("model preflight completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
