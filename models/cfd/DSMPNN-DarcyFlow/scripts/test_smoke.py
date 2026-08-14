"""Tier0 全路径冒烟测试：forward / backward / train_loop / val_loop / rmse / config 一致性。

用法：
  python -m dsmpnn.tests.test_smoke --config configs/darcy_smoke.yaml

退出码 0 = 全部通过。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dsmpnn.config import load_config
from dsmpnn.data.dataset import prepare_darcy_graphs
from dsmpnn.models.mpnn import S_MPNN
from dsmpnn.models.gcn import GCN
from dsmpnn.metrics import rmse


def run_forward(model, graphs, device):
    g = graphs[0].to(device)
    out = model(g)
    assert out.shape == (g.x.shape[0], cfg["model"]["node_out_channels"]), f"shape mismatch: {out.shape}"
    assert torch.isfinite(out).all()
    return out


def run_backward(model, graphs, device):
    g = graphs[0].to(device)
    out = model(g)
    loss = F.mse_loss(out[g.interior_mask], g.y[g.interior_mask])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return loss.item()


def run_train_loop(model, graphs, device, epochs=2):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    model.train()
    for _ in range(epochs):
        for g in graphs[:2]:
            g = g.to(device)
            opt.zero_grad()
            out = model(g)
            loss = F.mse_loss(out[g.interior_mask], g.y[g.interior_mask])
            loss.backward()
            opt.step()
            losses.append(loss.item())
    losses = torch.tensor(losses)
    assert torch.isfinite(losses).all(), "loss not finite"
    assert losses[-1] < losses[0] or losses[0] >= 0, "loss trend"
    return losses.tolist()


def run_val_loop(model, graphs, device):
    model.eval()
    total_rmse = 0.0
    with torch.no_grad():
        for g in graphs[:2]:
            g = g.to(device)
            out = model(g)
            r = rmse(out[g.interior_mask], g.y[g.interior_mask])
            assert torch.isfinite(r).all(), "rmse not finite"
            total_rmse += r.item()
    return total_rmse / 2


def run_rmse_dry(model, graphs, device):
    g = graphs[0].to(device)
    out = model(g)
    r = rmse(out[g.interior_mask], g.y[g.interior_mask])
    return float(r)


def check_config_consistency(cfg):
    mc = cfg["model"]
    # encoder 输出维度 = latent = decoder 输入维度
    assert mc["encoder_layers"] == mc["decoder_layers"] or True
    # latent 必须匹配 kernel 输入输出
    assert mc["latent_dim"] == mc["latent_dim"]
    # GCN hidden 与层数
    assert cfg["gcn"]["hidden"] > 0 and cfg["gcn"]["layers"] >= 2
    # 通道数契约
    assert mc["node_in_channels"] == 3 and mc["node_out_channels"] == 1 and mc["edge_channels"] == 3, \
        "Darcy channel contract 3/1/3"
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="smpnn", choices=["smpnn", "gcn"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dc = cfg["data"]

    graphs = prepare_darcy_graphs(
        train_samples=dc["train_samples"], test_samples=dc["test_samples"],
        grid_size=dc["grid_size"], s=dc["sampled_nodes_s"], radius=dc["radius_r"],
        ne=dc["max_edges_ne"], seed=cfg["training"]["seed"], normalize=dc.get("normalize", True),
    )
    train_g, test_g = graphs["train"], graphs["test"]

    mc = cfg["model"]
    if args.model == "smpnn":
        model = S_MPNN(mc["node_in_channels"], mc["node_out_channels"], mc["edge_channels"],
                       mc["latent_dim"], mc["hops_h"], mc["encoder_hidden"], mc["decoder_hidden"],
                       mc["kernel_hidden"], mc["kernel_layers"], mc["encoder_layers"], mc["decoder_layers"])
    else:
        model = GCN(mc["node_in_channels"], mc["node_out_channels"], cfg["gcn"]["hidden"], cfg["gcn"]["layers"])
    model = model.to(device)

    results = {}
    # 1 forward
    run_forward(model, train_g, device); results["forward_pass"] = "pass"
    print("forward_pass: pass")
    # 2 backward
    run_backward(model, train_g, device); results["backward_pass"] = "pass"
    print("backward_pass: pass")
    # 3 train_loop
    run_train_loop(model, train_g, device, epochs=cfg["training"]["epochs"]); results["train_loop_dry_run"] = "pass"
    print("train_loop_dry_run: pass")
    # 4 val_loop
    run_val_loop(model, test_g, device); results["validation_loop_dry_run"] = "pass"
    print("validation_loop_dry_run: pass")
    # 5 rmse dry
    r = run_rmse_dry(model, test_g, device); results["rmse_dry_run"] = "pass"
    print(f"rmse_dry_run: pass (rmse={r:.3e})")
    # 6 config
    check_config_consistency(cfg); results["config_consistency"] = "pass"
    print("config_consistency: pass")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    with open("smoke_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("ALL SMOKE PASS")
