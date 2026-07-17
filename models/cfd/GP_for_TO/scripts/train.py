import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (
    PROBLEMS,
    build_models,
    dump_json,
    ensure_onescience_path,
    load_config,
    resolve_path,
    save_checkpoint,
    select_device,
)

ensure_onescience_path()
from scripts.topology_optimization import find_TO


def parse_args():
    parser = argparse.ArgumentParser(description="Train GP_for_TO topology optimization models.")
    parser.add_argument("--problem", choices=PROBLEMS, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-iter", type=int, default=None)
    parser.add_argument("--n-col-domain", type=int, default=None)
    parser.add_argument("--n-train-per-bc", type=int, default=None)
    parser.add_argument("--diff-method", choices=("Numerical", "Autograd"), default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    if args.problem:
        cfg["problem"] = args.problem
    if args.gpu is not None:
        cfg["training"]["gpu"] = args.gpu
    if args.device:
        cfg["training"]["device"] = args.device
    if args.num_iter is not None:
        cfg["training"]["num_iter"] = args.num_iter
    if args.n_col_domain is not None:
        cfg["data"]["n_col_domain"] = args.n_col_domain
    if args.n_train_per_bc is not None:
        cfg["data"]["n_train_per_bc"] = args.n_train_per_bc
    if args.diff_method:
        cfg["training"]["diff_method"] = args.diff_method
    if args.lr is not None:
        cfg["training"]["lr_default"] = args.lr
    if args.checkpoint_path:
        cfg["training"]["checkpoint_path"] = args.checkpoint_path
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.no_plot:
        cfg["training"]["plot_outputs"] = False

    os.chdir(PROJECT_ROOT)
    device = select_device(cfg["training"])
    models, metadata = build_models(
        cfg,
        device,
        n_col_domain=cfg["data"]["n_col_domain"],
        n_train_per_bc=cfg["data"]["n_train_per_bc"],
        problem=cfg["problem"],
    )

    title = f"seed{cfg['seed']}_{cfg['problem']}_{datetime.now().strftime('%B%d_%H-%M')}"
    start_time = time.time()
    loss_history = find_TO(
        model_list=models,
        num_iter=int(cfg["training"]["num_iter"]),
        lr_default=float(cfg["training"]["lr_default"]),
        title=title,
        problem=cfg["problem"],
        diff_method=cfg["training"]["diff_method"],
        checkpoint_steps=list(cfg["training"].get("checkpoints", [])),
        plot_outputs=bool(cfg["training"].get("plot_outputs", True)),
    )
    elapsed = time.time() - start_time

    output_dir = resolve_path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "loss_history.npy", np.asarray(loss_history, dtype=np.float64))
    summary = {
        **metadata,
        "device": str(device),
        "num_iter": int(cfg["training"]["num_iter"]),
        "diff_method": cfg["training"]["diff_method"],
        "elapsed_seconds": elapsed,
        "final_loss": float(loss_history[-1]) if loss_history else None,
    }
    dump_json(summary, output_dir / "training_summary.json")
    ckpt_path = save_checkpoint(cfg["training"]["checkpoint_path"], models, cfg, summary, loss_history)

    print(f"Training finished in {elapsed:.2f}s")
    print(f"Loss history: {output_dir / 'loss_history.npy'}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
