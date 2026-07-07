#!/usr/bin/env python3
"""Preflight checks for the standardized Vortex shedding MGN runtime package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


EXPECTED_STATS = {
    "edge_stats.json": {"edge_mean": 3, "edge_std": 3},
    "node_stats.json": {
        "velocity_mean": 2,
        "velocity_std": 2,
        "velocity_diff_mean": 2,
        "velocity_diff_std": 2,
        "pressure_mean": 1,
        "pressure_std": 1,
    },
}

EXPECTED_FEATURES = {
    "cells": {"dtype": "int32", "shape": [1, -1, 3]},
    "mesh_pos": {"dtype": "float32", "shape": [1, -1, 2]},
    "node_type": {"dtype": "int32", "shape": [1, -1, 1]},
    "velocity": {"dtype": "float32", "shape": [600, -1, 2]},
    "pressure": {"dtype": "float32", "shape": [600, -1, 1]},
}


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="model package root")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("ONESCIENCE_CYLINDER_FLOW_DATA_DIR", "data/cylinder_flow"),
        help="directory containing meta.json, TFRecord splits and stats/",
    )
    parser.add_argument(
        "--skip-tfrecord-read",
        action="store_true",
        help="skip optional TensorFlow first-record readability check",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_runtime_files(repo_root: Path) -> dict:
    required = [
        "train.py",
        "inference.py",
        "conf/mgn_cylinderflow.yaml",
        "scripts/preflight_vortex_shedding_mgn.py",
    ]
    for rel_path in required:
        if not (repo_root / rel_path).is_file():
            fail(f"missing runtime file: {rel_path}")
    cfg = load_yaml(repo_root / "conf" / "mgn_cylinderflow.yaml")
    datapipe = cfg.get("datapipe", {})
    data = datapipe.get("data", {})
    source = datapipe.get("source", {})
    expected_counts = {
        "train_samples": 400,
        "train_steps": 300,
        "val_samples": 10,
        "val_steps": 300,
        "test_samples": 10,
        "test_steps": 300,
    }
    for key, value in expected_counts.items():
        if data.get(key) != value:
            fail(f"config datapipe.data.{key} expected {value}, got {data.get(key)}")
    if source.get("data_dir") != "data/cylinder_flow":
        fail("config datapipe.source.data_dir must be data/cylinder_flow")
    if source.get("stats_dir") != "data/cylinder_flow/stats":
        fail("config datapipe.source.stats_dir must be data/cylinder_flow/stats")
    model = cfg.get("model", {})
    if model.get("name") != "MeshGraphNet":
        fail("config model.name must be MeshGraphNet")
    params = model.get("specific_params", {}).get("MeshGraphNet", {})
    if params.get("num_input_features") != 6 or params.get("num_edge_features") != 3:
        fail("MeshGraphNet input feature dimensions do not match cylinder_flow schema")
    if params.get("num_output_features") != 3:
        fail("MeshGraphNet output feature dimension must be 3")
    return cfg


def check_dataset_files(data_dir: Path) -> dict:
    if not data_dir.is_dir():
        fail(f"missing data directory: {data_dir}")
    for name in ["train.tfrecord", "valid.tfrecord", "test.tfrecord", "meta.json"]:
        path = data_dir / name
        if not path.is_file():
            fail(f"missing data file: {path}")
        if path.stat().st_size <= 0:
            fail(f"empty data file: {path}")
    stats_dir = data_dir / "stats"
    if not stats_dir.is_dir():
        fail(f"missing stats directory: {stats_dir}")
    for name, schema in EXPECTED_STATS.items():
        path = stats_dir / name
        if not path.is_file():
            fail(f"missing stats file: {path}")
        stats = json.loads(path.read_text(encoding="utf-8"))
        for key, length in schema.items():
            value = stats.get(key)
            if not isinstance(value, list) or len(value) != length:
                fail(f"{name}.{key} must be a length-{length} list")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("trajectory_length") != 600:
        fail(f"meta trajectory_length expected 600, got {meta.get('trajectory_length')}")
    if meta.get("simulator") != "comsol":
        fail(f"meta simulator expected comsol, got {meta.get('simulator')}")
    if meta.get("field_names") != ["cells", "mesh_pos", "node_type", "velocity", "pressure"]:
        fail("meta field_names do not match expected cylinder_flow fields")
    features = meta.get("features", {})
    for name, expected in EXPECTED_FEATURES.items():
        feature = features.get(name)
        if not feature:
            fail(f"meta missing feature: {name}")
        if feature.get("dtype") != expected["dtype"] or feature.get("shape") != expected["shape"]:
            fail(f"meta feature {name} schema mismatch")
    return meta


def check_tfrecord_first_record(data_dir: Path, meta: dict) -> None:
    try:
        import numpy as np
        import tensorflow.compat.v1 as tf
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        print(f"[WARN] TensorFlow first-record check skipped: {exc}")
        return

    feature_dict = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    for split in ["train", "valid", "test"]:
        record_iter = iter(tf.data.TFRecordDataset(str(data_dir / f"{split}.tfrecord")).take(1))
        try:
            raw = next(record_iter)
        except StopIteration:
            fail(f"{split}.tfrecord contains no records")
        features = tf.io.parse_single_example(raw, feature_dict)
        decoded = {}
        for key, spec in meta["features"].items():
            values = features[key].values
            if len(values) == 0:
                fail(f"{split}.tfrecord first record missing feature {key}")
            decoded[key] = np.frombuffer(values[0].numpy(), dtype=getattr(np, spec["dtype"]))
        if decoded["velocity"].size % (meta["trajectory_length"] * 2) != 0:
            fail(f"{split}.tfrecord velocity size incompatible with trajectory_length")
        nodes = decoded["velocity"].size // (meta["trajectory_length"] * 2)
        if decoded["pressure"].size != meta["trajectory_length"] * nodes:
            fail(f"{split}.tfrecord pressure node count mismatch")
        if decoded["mesh_pos"].size != nodes * 2:
            fail(f"{split}.tfrecord mesh_pos node count mismatch")
        if decoded["node_type"].size != nodes:
            fail(f"{split}.tfrecord node_type count mismatch")
        if decoded["cells"].size % 3 != 0:
            fail(f"{split}.tfrecord cells are not triangular")
        print(
            f"[OK] {split}.tfrecord first record: "
            f"nodes={nodes}, cells={decoded['cells'].size // 3}, "
            "velocity_dtype=float32, pressure_dtype=float32"
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    check_runtime_files(repo_root)
    meta = check_dataset_files(data_dir)
    if not args.skip_tfrecord_read:
        check_tfrecord_first_record(data_dir, meta)
    print("[OK] Vortex shedding MGN model preflight passed")


if __name__ == "__main__":
    main()
