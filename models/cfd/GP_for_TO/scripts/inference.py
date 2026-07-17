import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    PROBLEMS,
    build_models,
    ensure_onescience_path,
    load_checkpoint,
    load_config,
    resolve_path,
    select_device,
    tensor_to_numpy_dict,
)

ensure_onescience_path()
from scripts.topology_optimization import clear_cached_kernels, predict_fields, share_mean_module


def parse_args():
    parser = argparse.ArgumentParser(description="Run GP_for_TO field inference from a checkpoint.")
    parser.add_argument("--problem", choices=PROBLEMS, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-col-domain", type=int, default=None)
    parser.add_argument("--n-train-per-bc", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    if args.problem:
        cfg["problem"] = args.problem
    if args.gpu is not None:
        cfg["inference"]["gpu"] = args.gpu
    if args.device:
        cfg["inference"]["device"] = args.device
    if args.n_col_domain is not None:
        cfg["inference"]["n_col_domain"] = args.n_col_domain
    if args.n_train_per_bc is not None:
        cfg["data"]["n_train_per_bc"] = args.n_train_per_bc
    if args.checkpoint_path:
        cfg["inference"]["checkpoint_path"] = args.checkpoint_path
    if args.output_dir:
        cfg["inference"]["output_dir"] = args.output_dir

    os.chdir(PROJECT_ROOT)
    device = select_device(cfg["inference"])
    models, metadata = build_models(
        cfg,
        device,
        n_col_domain=cfg["inference"].get("n_col_domain", cfg["data"]["n_col_domain"]),
        n_train_per_bc=cfg["data"]["n_train_per_bc"],
        problem=cfg["problem"],
    )
    checkpoint = load_checkpoint(cfg["inference"]["checkpoint_path"], models, device)
    share_mean_module(models)
    for model in models:
        model.eval()
    clear_cached_kernels(models)

    with torch.no_grad():
        fields = predict_fields(models)

    output_dir = resolve_path(cfg["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "predictions.npz"
    np.savez(npz_path, **tensor_to_numpy_dict(fields))

    summary = {
        **metadata,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "output_file": str(npz_path),
        "field_shapes": {key: list(value.shape) for key, value in fields.items()},
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved predictions to {npz_path}")
    print(json.dumps(summary["field_shapes"], indent=2))


if __name__ == "__main__":
    main()
