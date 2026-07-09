#!/usr/bin/env python3
"""Run Protenix finetuning with package-local defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

os.chdir(PACKAGE_ROOT)
os.environ.setdefault("DATA_ROOT_DIR", "../bio_protenix_dataset")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

DEFAULT_ARGS = [
    "--run_name",
    "protenix_finetune",
    "--seed",
    "42",
    "--base_dir",
    "./output",
    "--dtype",
    "bf16",
    "--project",
    "protenix",
    "--use_wandb",
    "false",
    "--diffusion_batch_size",
    "48",
    "--eval_interval",
    "400",
    "--log_interval",
    "50",
    "--checkpoint_interval",
    "400",
    "--ema_decay",
    "0.999",
    "--train_crop_size",
    "384",
    "--max_steps",
    "100000",
    "--warmup_steps",
    "2000",
    "--lr",
    "0.001",
    "--sample_diffusion.N_step",
    "20",
    "--load_checkpoint_path",
    "./weight/model_v0.5.0.pt",
    "--load_ema_checkpoint_path",
    "./weight/model_v0.5.0.pt",
    "--data.train_sets",
    "weightedPDB_before2109_wopb_nometalc_0925",
    "--data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list",
    "ft_datasets/finetune_subset.txt",
    "--data.test_sets",
    "recentPDB_1536_sample384_0925,posebusters_0925",
]

sys.argv[1:] = DEFAULT_ARGS + sys.argv[1:]

from scripts.runner.train import main


if __name__ == "__main__":
    main()
