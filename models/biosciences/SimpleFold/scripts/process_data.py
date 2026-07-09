#!/usr/bin/env python
#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import argparse
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.datapipes.simplefold.process_mmcif import process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MMCIF data.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="The directory containing the MMCIF files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default="data",
        help="The output directory.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The number of processes.",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="The Redis host, only used with --use-redis.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=7777,
        help="The Redis port, only used with --use-redis.",
    )
    parser.add_argument(
        "--use-redis",
        action="store_true",
        help="Use an external Redis CCD resource instead of local weight/ccd.pkl.",
    )
    parser.add_argument(
        "--ccd-path",
        type=Path,
        default=ROOT / "weight" / "ccd.pkl",
        help="Path to the local CCD pickle file.",
    )
    parser.add_argument(
        "--use-assembly",
        action="store_true",
        help="Whether to use assembly 1.",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    process(args)
