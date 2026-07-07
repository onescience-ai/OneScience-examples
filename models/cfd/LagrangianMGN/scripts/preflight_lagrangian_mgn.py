#!/usr/bin/env python3
"""Preflight checks for the standardized Lagrangian MGN runtime package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


EXPECTED_METADATA_KEYS = {
    "bounds",
    "sequence_length",
    "default_connectivity_radius",
    "dim",
    "dt",
    "vel_mean",
    "vel_std",
    "acc_mean",
    "acc_std",
}


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def read_text(path: Path) -> str:
    if not path.is_file():
        fail(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="model package root")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("ONESCIENCE_LAGRANGIAN_DATA_DIR", "data/Water"),
        help="Water data directory containing metadata.json and TFRecord splits",
    )
    parser.add_argument(
        "--skip-tfrecord-read",
        action="store_true",
        help="skip optional TensorFlow first-record readability check",
    )
    return parser.parse_args()


def check_config(repo_root: Path) -> None:
    config = read_text(repo_root / "conf" / "config.yaml")
    water = read_text(repo_root / "conf" / "experiment" / "water.yaml")
    required_files = [
        "train.py",
        "inference.py",
        "loggers.py",
        "conf/data/lagrangian_dataset.yaml",
        "conf/model/mgn.yaml",
        "conf/model/mgn_2d.yaml",
        "conf/loss/mseloss.yaml",
        "conf/optimizer/adam.yaml",
        "conf/lr_scheduler/cosine.yaml",
    ]
    for rel_path in required_files:
        if not (repo_root / rel_path).is_file():
            fail(f"missing runtime file: {rel_path}")

    expected_fragments = [
        "data_dir: ${oc.env:ONESCIENCE_LAGRANGIAN_DATA_DIR,data/Water}",
        'name: "Water"',
        "num_history: 5",
        "num_node_types: 6",
    ]
    for fragment in expected_fragments:
        if fragment not in config:
            fail(f"conf/config.yaml missing expected fragment: {fragment}")
    if "/model: mgn_2d" not in water or "name: Water" not in water:
        fail("conf/experiment/water.yaml is not configured for 2D Water")


def check_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.json"
    if not meta_path.is_file():
        fail(f"missing metadata file: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = sorted(EXPECTED_METADATA_KEYS - set(metadata))
    if missing:
        fail(f"metadata.json missing keys: {missing}")
    if metadata["dim"] != 2:
        fail(f"expected Water dim=2, got {metadata['dim']}")
    if metadata["sequence_length"] != 1000:
        fail(f"expected Water sequence_length=1000, got {metadata['sequence_length']}")
    for key in ["vel_mean", "vel_std", "acc_mean", "acc_std"]:
        if len(metadata[key]) != metadata["dim"]:
            fail(f"metadata {key} length does not match dim")
    if len(metadata["bounds"]) != metadata["dim"]:
        fail("metadata bounds length does not match dim")
    return metadata


def check_split_files(data_dir: Path) -> None:
    for split in ["train", "valid", "test"]:
        path = data_dir / f"{split}.tfrecord"
        if not path.is_file():
            fail(f"missing split file: {path}")
        if path.stat().st_size <= 0:
            fail(f"empty split file: {path}")


def check_tfrecord_first_record(data_dir: Path, metadata: dict) -> None:
    try:
        import numpy as np
        import tensorflow.compat.v1 as tf
    except Exception as exc:  # pragma: no cover - depends on runtime env
        print(f"[WARN] TensorFlow first-record check skipped: {exc}")
        return

    feature_description = {"position": tf.io.VarLenFeature(tf.string)}
    context_features = {
        "key": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "particle_type": tf.io.VarLenFeature(tf.string),
    }
    expected_steps = metadata["sequence_length"] + 1
    dim = metadata["dim"]
    for split in ["train", "valid", "test"]:
        record_iter = iter(tf.data.TFRecordDataset(str(data_dir / f"{split}.tfrecord")).take(1))
        try:
            raw = next(record_iter)
        except StopIteration:
            fail(f"{split}.tfrecord contains no records")
        context, features = tf.io.parse_single_sequence_example(
            raw,
            context_features=context_features,
            sequence_features=feature_description,
        )
        position_values = features["position"].values
        if len(position_values) == 0:
            fail(f"{split}.tfrecord first record has no position feature")
        position = np.frombuffer(position_values[0].numpy(), dtype=np.float32)
        if position.size % (expected_steps * dim) != 0:
            fail(f"{split}.tfrecord first record position size is incompatible with metadata")
        particle_type_values = context["particle_type"].values
        if len(particle_type_values) == 0:
            fail(f"{split}.tfrecord first record has no particle_type")
        particle_type = np.frombuffer(particle_type_values[0].numpy(), dtype=np.int64)
        num_particles = position.size // (expected_steps * dim)
        if particle_type.shape[0] != num_particles:
            fail(f"{split}.tfrecord particle_type length does not match position particles")
        print(
            f"[OK] {split}.tfrecord first record: "
            f"position_shape=({expected_steps}, {num_particles}, {dim}), "
            "position_dtype=float32, particle_type_dtype=int64"
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    check_config(repo_root)
    check_split_files(data_dir)
    metadata = check_metadata(data_dir)
    if not args.skip_tfrecord_read:
        check_tfrecord_first_record(data_dir, metadata)
    print("[OK] Lagrangian MGN preflight passed")


if __name__ == "__main__":
    main()
