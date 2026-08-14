"""Training entry point for Packed-Ensemble MLP on AirfRANS.

Reproduces arXiv:2312.13403 PE(8,4,1): layers=(64,64,8,64,64,64,8,64,64),
M=8, alpha=4, gamma=1, Adam lr=2e-4, weight_decay=1e-5, MSE loss.

Usage:
    python train.py --model pe_mlp --layers "64,64,8,64,64,64,8,64,64" \
        --num-estimators 8 --alpha 4 --gamma 1 --lr 2e-4 --wd 1e-5 \
        --epochs 30 --data-dir <DATA> --out-dir <OUT> [--dcu]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset import AirfRANSDataset, get_splits, list_simulations
from model import model_factory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pe_mlp", choices=["pe_mlp", "mlp"])
    p.add_argument("--layers", default="64,64,8,64,64,64,8,64,64")
    p.add_argument("--in-features", type=int, default=7)
    p.add_argument("--out-features", type=int, default=4)
    p.add_argument("--num-estimators", type=int, default=8)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--gamma", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-nodes-per-sim", type=int, default=8000)
    p.add_argument("--n-train-sims", type=int, default=12)
    p.add_argument("--n-eval-sims", type=int, default=6)
    p.add_argument("--data-dir", required=True,
                   default="/public/home/wangqi_scnet/batch_paper_repo/data/airfrans/data/Dataset")
    p.add_argument("--out-dir", default="repro/2312.13403/output")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dcu", action="store_true", help="use DCU/AMD GPU (HIP_VISIBLE_DEVICES)")
    p.add_argument("--save-stats", action="store_true", default=True)
    return p.parse_args()


def _make_model(args):
    layers = tuple(int(x) for x in args.layers.split(","))
    model = model_factory(
        args.model,
        in_features=args.in_features,
        out_features=args.out_features,
        layers=layers,
        num_estimators=args.num_estimators,
        alpha=args.alpha,
        gamma=args.gamma,
        dropout=args.dropout,
    )
    return model


def prepare_data(args, device):
    from dataset import compute_stats, list_simulations, get_splits
    from sklearn.preprocessing import StandardScaler
    import torch

    sims = list_simulations(args.data_dir)
    splits = get_splits(args.data_dir)
    train_sims = splits["train"][: args.n_train_sims]
    test_sims = splits["test"][: args.n_eval_sims]
    ood_sims = splits["test_ood"][: args.n_eval_sims]

    train_ds = AirfRANSDataset(args.data_dir, train_sims, max_nodes_per_sim=args.max_nodes_per_sim)
    scaler_in = StandardScaler().fit(train_ds.x)
    scaler_out = StandardScaler().fit(train_ds.y)

    tx = scaler_in.transform(train_ds.x).astype(np.float32)
    ty = scaler_out.transform(train_ds.y).astype(np.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )

    def make_eval(sims):
        d = AirfRANSDataset(args.data_dir, sims, max_nodes_per_sim=-1)
        ex = scaler_in.transform(d.x).astype(np.float32)
        ey = scaler_out.transform(d.y).astype(np.float32)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(ex), torch.from_numpy(ey))
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers)

    eval_loaders = {
        "val": make_eval(train_sims[args.n_train_sims // 2:]),  # holdout subset
        "test": make_eval(test_sims),
        "test_ood": make_eval(ood_sims),
    }
    stats = {
        "scaler_in_mean": scaler_in.mean_.tolist(),
        "scaler_in_scale": scaler_in.scale_.tolist(),
        "scaler_out_mean": scaler_out.mean_.tolist(),
        "scaler_out_scale": scaler_out.scale_.tolist(),
        "train_sims": train_sims,
        "test_sims": test_sims,
        "ood_sims": ood_sims,
    }
    return train_loader, eval_loaders, stats, scaler_out


@torch.no_grad()
def evaluate(model, loader, device, metric_fn):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        # pe_mlp returns (B, M, N, out); average ensemble
        if out.dim() == 4:
            out = out.mean(dim=1)  # (B, N, out)
        elif out.dim() == 3 and model.M and hasattr(model, "M") and out.shape[1] == model.M:
            out = out.mean(dim=1)
        out = out.reshape(yb.shape)
        preds.append(out.detach().cpu())
        trues.append(yb.cpu())
    return metric_fn(preds, trues)


def mse_loss_metric(preds, trues):
    pred = torch.cat(preds, 0)
    true = torch.cat(trues, 0)
    return float(((pred - true) ** 2).mean().item())


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    if args.dcu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("[warn] DCU requested but torch.cuda not available; using CPU")
    print(f"[info] device={device}")

    model = _make_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model={args.model} params={n_params}")
    total_train = sum(
        p.numel() for n, p in model.named_parameters() if "weight" in n
    )
    print(f"[info] total weight params={total_train}")

    train_loader, eval_loaders, stats, scaler_out = prepare_data(args, device)
    print(f"[info] train samples ~ {sum(len(d) for d, _ in [])}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "test_loss": [], "test_ood_loss": []}
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            if out.dim() == 4:
                out = out.mean(dim=1)  # (B, N, out)
            out = out.reshape(yb.shape)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        val_loss = evaluate(model, eval_loaders["val"], device, mse_loss_metric)
        test_loss = evaluate(model, eval_loaders["test"], device, mse_loss_metric)
        ood_loss = evaluate(model, eval_loaders["test_ood"], device, mse_loss_metric)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["test_ood_loss"].append(ood_loss)
        print(
            f"[epoch {epoch+1}/{args.epochs}] train={train_loss:.6f} "
            f"val={val_loss:.6f} test={test_loss:.6f} ood={ood_loss:.6f} "
            f"({time.time()-t0:.1f}s)"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model,
                    "config": vars(args),
                    "stats": stats,
                },
                os.path.join(args.out_dir, "model_best.pt"),
            )

    # save final
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": args.model,
            "config": vars(args),
            "stats": stats,
        },
        os.path.join(args.out_dir, "model_final.pt"),
    )
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[done] saved to {args.out_dir}")


if __name__ == "__main__":
    main()
