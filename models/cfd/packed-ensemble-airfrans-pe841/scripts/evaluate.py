"""Evaluation for PE-MLP: MSE + physics metrics (drag/lift/Spearman).

Usage:
    python evaluate.py --checkpoint <model_final.pt> --data-dir <DATA> \
        --split test --out <out.json> [--dcu]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset import AirfRANSDataset
from metrics import summarize
from model import model_factory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True,
                   default="/public/home/wangqi_scnet/batch_paper_repo/data/airfrans/data/Dataset")
    p.add_argument("--split", default="test", choices=["test", "test_ood", "val"])
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--dcu", action="store_true")
    p.add_argument("--out", default="repro/2312.13403/output/eval.json")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    stats = ckpt["stats"]
    model_type = ckpt["model_type"]

    device = torch.device("cuda" if args.dcu and torch.cuda.is_available() else "cpu")
    model = model_factory(
        model_type,
        in_features=cfg["in_features"],
        out_features=cfg["out_features"],
        layers=tuple(int(x) for x in cfg["layers"].split(",")),
        num_estimators=cfg["num_estimators"],
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    stats_key = {"test": "test_sims", "test_ood": "ood_sims", "val": "train_sims"}[args.split]
    sim_names = stats[stats_key]
    scaler_out_mean = np.array(stats["scaler_out_mean"], dtype=np.float32)
    scaler_out_scale = np.array(stats["scaler_out_scale"], dtype=np.float32)
    scaler_in_mean = np.array(stats["scaler_in_mean"], dtype=np.float32)
    scaler_in_scale = np.array(stats["scaler_in_scale"], dtype=np.float32)

    preds_scaled, true_scaled, inlet_vels = [], [], []
    for sim in sim_names:
        d = AirfRANSDataset(args.data_dir, [sim], max_nodes_per_sim=-1)
        x = torch.from_numpy((d.x - scaler_in_mean) / scaler_in_scale).float().to(device)
        y = torch.from_numpy(d.y).float()
        with torch.no_grad():
            out = model(x)
            if out.dim() == 4:
                out = out.mean(dim=1)
            out = out.reshape(-1, 4)
        preds_scaled.append(out.cpu().numpy())
        true_scaled.append(((y.numpy() - scaler_out_mean) / scaler_out_scale))
        # inlet velocity: store per-sim u_inf (from data, approximated from feature column)
        # reconstruct u_inf from standardized input
        u_inf = d.x[0, 2:4]  # u_inf_x, u_inf_y raw
        inlet_vels.append(u_inf)

    result = summarize(
        args.data_dir,
        sim_names,
        preds_scaled,
        true_scaled,
        scaler_out_mean,
        scaler_out_scale,
        inlet_vels,
    )
    result["split"] = args.split
    result["n_sims"] = len(sim_names)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
