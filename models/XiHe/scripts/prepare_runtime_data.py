#!/usr/bin/env python3
"""Prepare CMEMS files for the XiHe runtime package."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


YEARS = [1993, 1994, 1995, 1996, 1997, 1998, 1999]


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return
        raise RuntimeError(f"target exists with different size: {dst}")
    if copy:
        shutil.copy2(src, dst)
    else:
        os.link(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize downloaded OneScience/CMEMS files for XiHe."
    )
    parser.add_argument(
        "--dataset-root",
        default="../CMEMS",
        help="Downloaded OneScience/CMEMS dataset root; it must contain data/*.h5.",
    )
    parser.add_argument(
        "--runtime-data-dir",
        default="data",
        help="XiHe runtime data directory; config.yaml defaults to ./data/.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating hard links.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    runtime_data_dir = Path(args.runtime_data_dir).resolve()
    source_data_dir = dataset_root / "data"
    target_h5_dir = runtime_data_dir / "data"
    stats_dir = runtime_data_dir / "stats"

    missing = [str(source_data_dir / f"{year}.h5") for year in YEARS if not (source_data_dir / f"{year}.h5").exists()]
    if missing:
        raise FileNotFoundError("missing CMEMS yearly files: " + ", ".join(missing))

    for year in YEARS:
        link_or_copy(source_data_dir / f"{year}.h5", target_h5_dir / f"{year}.h5", args.copy)

    first_file = source_data_dir / f"{YEARS[0]}.h5"
    with h5py.File(first_file, "r") as handle:
        means = handle["global_means"][()]
        stds = handle["global_stds"][()]
        fields_shape = handle["fields"].shape

    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "global_means.npy", means.astype("float32", copy=False))
    np.save(stats_dir / "global_stds.npy", stds.astype("float32", copy=False))

    mask_path = runtime_data_dir / "land_mask.npy"
    if not mask_path.exists():
        np.save(mask_path, np.ones(fields_shape[-2:], dtype=np.float32))

    print(f"prepared CMEMS data under {runtime_data_dir}")
    print(f"yearly HDF5 files: {target_h5_dir}")
    print(f"stats: {stats_dir}")
    print(f"mask: {mask_path}")


if __name__ == "__main__":
    main()
