#!/usr/bin/env python3
"""Run the legacy Protenix inference entrypoint with local defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

os.chdir(PACKAGE_ROOT)
os.environ.setdefault("DATA_ROOT_DIR", "../bio_protenix_dataset")

DEFAULT_ARGS = [
    "--seeds",
    "101",
    "--dtype",
    "bf16",
    "--num_workers",
    "8",
    "--load_checkpoint_path",
    "./weight/model_v0.5.0.pt",
    "--dump_dir",
    "./output",
    "--input_json_path",
    "./examples/7r6r.json",
    "--model.N_cycle",
    "10",
    "--sample_diffusion.N_sample",
    "5",
    "--sample_diffusion.N_step",
    "200",
    "--use_msa",
    "true",
]

if len(sys.argv) == 1:
    sys.argv.extend(DEFAULT_ARGS)

from scripts.runner.inference import run


if __name__ == "__main__":
    run()
