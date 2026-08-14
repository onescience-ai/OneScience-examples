"""训练入口：S-MPNN（单卡）与 DS-MPNN（分布式）。

用法：
  python -m dsmpnn.train --config configs/darcy.yaml --model smpnn
  python -m dsmpnn.train --config configs/darcy.yaml --model dsmpnn --nproc 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from dsmpnn.config import load_config
from dsmpnn.data.dataset import prepare_darcy_graphs, save_graphs
from dsmpnn.models.mpnn import S_MPNN
from dsmpnn.models.ds_mpnn import DS_MPNN
from dsmpnn.models.gcn import GCN
from dsmpnn.metrics import rmse


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, model_type: str) -> torch.nn.Module:
    mc = cfg["model"]
    gc = cfg["gcn"]
    if model_type == "smpnn":
        return S_MPNN(
            node_in_channels=mc["node_in_channels"],
            node_out_channels=mc["node_out_channels"],
            edge_channels=mc["edge_channels"],
            latent_dim=mc["latent_dim"],
            hops=mc["hops_h"],
            encoder_hidden=mc["encoder_hidden"],
            decoder_hidden=mc["decoder_hidden"],
            kernel_hidden=mc["kernel_hidden"],
            kernel_layers=mc["kernel_layers"],
            encoder_layers=mc["encoder_layers"],
            decoder_layers=mc["decoder_layers"],
        )
    elif model_type == "dsmpnn":
        base = S_MPNN(
            node_in_channels=mc["node_in_channels"],
            node_out_channels=mc["node_out_channels"],
            edge_channels=mc["edge_channels"],
            latent_dim=mc["latent_dim"],
            hops=mc["hops_h"],
            encoder_hidden=mc["encoder_hidden"],
            decoder_hidden=mc["decoder_hidden"],
            kernel_hidden=mc["kernel_hidden"],
            kernel_layers=mc["kernel_layers"],
            encoder_layers=mc["encoder_layers"],
            decoder_layers=mc["decoder_layers"],
        )
        return DS_MPNN(base, use_communication=True)
    elif model_type == "gcn":
        return GCN(mc["node_in_channels"], mc["node_out_channels"], gc["hidden"], gc["layers"])
    else:
        raise ValueError(f"unknown model type: {model_type}")


def train_epoch(model, graphs, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for g in graphs:
        g = g.to(device)
        optimizer.zero_grad()
        out = model(g)
        target = g.y
        # interior mask loss（论文：仅 interior points 计损失）
        mask = g.interior_mask
        loss = F.mse_loss(out[mask], target[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate_loop(model, graphs, device):
    model.eval()
    total_rmse = 0.0
    total_l1 = 0.0
    n = 0
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            out = model(g)
            target = g.y
            mask = g.interior_mask
            total_rmse += rmse(out[mask], target[mask]).item()
            total_l1 += F.l1_loss(out[mask], target[mask]).item()
            n += 1
    return total_rmse / max(n, 1), total_l1 / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--model", default="smpnn", choices=["smpnn", "dsmpnn", "gcn"])
    parser.add_argument("--nproc", type=int, default=1, help="分布式 rank 数")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-graphs", action="store_true", help="保存构建的图数据")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    dc = cfg["data"]
    tc = cfg["training"]

    # 初始化分布式（如需）
    if args.model == "dsmpnn" and args.nproc > 1:
        dist.init_process_group(backend=cfg["distributed"].get("backend", "gloo"),
                                init_method="env://", rank=args.rank, world_size=args.nproc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备（支持缓存与多进程）
    graphs = prepare_darcy_graphs(
        train_samples=dc["train_samples"],
        test_samples=dc["test_samples"],
        grid_size=dc["grid_size"],
        s=dc["sampled_nodes_s"],
        radius=dc["radius_r"],
        ne=dc["max_edges_ne"],
        seed=tc["seed"],
        normalize=dc.get("normalize", True),
        nproc=dc.get("nproc", 1),
        cache_dir=dc.get("data_dir"),
    )
    train_graphs, test_graphs = graphs["train"], graphs["test"]
    if args.save_graphs:
        save_graphs(train_graphs, os.path.join(dc["data_dir"], "train"))
        save_graphs(test_graphs, os.path.join(dc["data_dir"], "test"))
        with open(os.path.join(dc["data_dir"], "stats.json"), "w") as f:
            json.dump(graphs["stats"], f)
        print(f"[data] saved {len(train_graphs)} train / {len(test_graphs)} test graphs to {dc['data_dir']}")

    model = build_model(cfg, args.model)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} params = {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tc["lr"], total_steps=tc["epochs"] * len(train_graphs),
        pct_start=0.1, div_factor=10.0, final_div_factor=100.0
    )

    epochs = args.epochs if args.epochs is not None else tc["epochs"]
    os.makedirs(tc["checkpoint_dir"], exist_ok=True)
    os.makedirs(tc["log_dir"], exist_ok=True)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_graphs, optimizer, device)
        for _ in range(len(train_graphs)):
            scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            val_rmse, val_l1 = validate_loop(model, test_graphs, device)
            print(f"epoch {epoch:3d}/{epochs} train_loss={loss:.6f} val_rmse={val_rmse:.3e} val_l1={val_l1:.3e}")

    val_rmse, val_l1 = validate_loop(model, test_graphs, device)
    print(f"[final] test_rmse={val_rmse:.3e} test_l1={val_l1:.3e}")

    # 保存 checkpoint
    ckpt_path = os.path.join(tc["checkpoint_dir"], f"{args.model}_final.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg, "model_type": args.model,
                "test_rmse": val_rmse, "n_params": n_params}, ckpt_path)
    print(f"[save] {ckpt_path}")

    if args.model == "dsmpnn" and args.nproc > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
