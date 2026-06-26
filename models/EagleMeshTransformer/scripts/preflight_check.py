#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


REQUIRED_KEYS = ["pointcloud", "mask", "VX", "VY", "PS", "PG"]
SPLITS = ["train", "valid", "test"]


def fail(message):
    raise SystemExit(f"[ERROR] {message}")


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return YAML(typ="safe").load(f)


def resolve_env_value(value):
    if not isinstance(value, str):
        return value
    out = value
    for key, env_value in os.environ.items():
        out = out.replace("${" + key + "}", env_value).replace("$" + key, env_value)
    return out


def inspect_sample(sample_dir, n_cluster):
    sim_path = sample_dir / "sim.npz"
    tri_path = sample_dir / "triangles.npy"
    cluster_path = sample_dir / f"constrained_kmeans_{n_cluster}.npy"
    for path in [sim_path, tri_path, cluster_path]:
        if not path.exists():
            fail(f"missing required data file: {path}")

    with np.load(sim_path, mmap_mode="r") as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            fail(f"{sim_path} missing keys: {missing}")
        pointcloud = data["pointcloud"]
        mask = data["mask"]
        if pointcloud.ndim != 3 or pointcloud.shape[-1] != 2:
            fail(f"unexpected pointcloud shape: {pointcloud.shape}")
        if mask.shape != pointcloud.shape[:2]:
            fail(f"mask shape {mask.shape} does not match pointcloud {pointcloud.shape[:2]}")
        for key in ["VX", "VY", "PS", "PG"]:
            if data[key].shape != pointcloud.shape[:2]:
                fail(f"{key} shape {data[key].shape} does not match pointcloud {pointcloud.shape[:2]}")
        if pointcloud.dtype != np.float32:
            fail(f"pointcloud dtype should be float32, got {pointcloud.dtype}")

    triangles = np.load(tri_path, mmap_mode="r")
    if triangles.ndim != 3 or triangles.shape[0] != pointcloud.shape[0] or triangles.shape[-1] != 3:
        fail(f"unexpected triangles shape: {triangles.shape}")

    clusters = np.load(cluster_path, mmap_mode="r")
    if clusters.ndim != 3 or clusters.shape[0] != pointcloud.shape[0] or clusters.shape[-1] != n_cluster:
        fail(f"unexpected cluster shape for n_cluster={n_cluster}: {clusters.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/graphvit_eagle.yaml")
    parser.add_argument("--sample-limit", type=int, default=2)
    args = parser.parse_args()

    root = Path.cwd()
    config_path = root / args.config
    if not config_path.exists():
        fail(f"config not found: {config_path}")

    cfg = load_yaml(config_path)
    source = cfg["datapipe"]["source"]
    data_dir = Path(resolve_env_value(source["data_dir"]))
    cluster_dir = Path(resolve_env_value(source["cluster_dir"]))
    splits_dir = root / source["splits_dir"]
    n_cluster = int(cfg["datapipe"]["data"]["n_cluster"])

    if "ONESCIENCE_EAGLE_DATA_DIR" not in os.environ:
        fail("ONESCIENCE_EAGLE_DATA_DIR is not set")
    if not data_dir.exists():
        fail(f"data_dir does not exist: {data_dir}")
    if not cluster_dir.exists():
        fail(f"cluster_dir does not exist: {cluster_dir}")
    if data_dir.resolve() != cluster_dir.resolve():
        fail(f"data_dir and cluster_dir should point to the same Eagle_dataset root: {data_dir} vs {cluster_dir}")
    if n_cluster not in [-1, 1, 10, 20, 30, 40]:
        fail(f"unsupported n_cluster: {n_cluster}")

    total = 0
    for split in SPLITS:
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            fail(f"missing split file: {split_file}")
        rels = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rels:
            fail(f"empty split file: {split_file}")
        for rel in rels:
            sample_dir = data_dir / rel
            if not sample_dir.exists():
                fail(f"split {split} points to missing sample directory: {sample_dir}")
        for rel in rels[: args.sample_limit]:
            inspect_sample(data_dir / rel, n_cluster)
        total += len(rels)
        print(f"[OK] split {split}: {len(rels)} entries, sampled {min(args.sample_limit, len(rels))}")

    print(f"[OK] model preflight completed: {total} split entries checked against {data_dir}")


if __name__ == "__main__":
    main()
