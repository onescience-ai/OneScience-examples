"""推理/评估入口：加载 checkpoint，计算 RMSE/L1，输出结果。

用法：
  python -m dsmpnn.evaluate --config configs/darcy.yaml --checkpoint checkpoints/smpnn_final.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from dsmpnn.config import load_config
from dsmpnn.data.dataset import prepare_darcy_graphs
from dsmpnn.models.mpnn import S_MPNN
from dsmpnn.models.ds_mpnn import DS_MPNN
from dsmpnn.models.gcn import GCN
from dsmpnn.metrics import rmse


def build_model_from_config(cfg, model_type):
    mc = cfg["model"]
    gc = cfg["gcn"]
    if model_type == "smpnn":
        return S_MPNN(mc["node_in_channels"], mc["node_out_channels"], mc["edge_channels"],
                      mc["latent_dim"], mc["hops_h"], mc["encoder_hidden"], mc["decoder_hidden"],
                      mc["kernel_hidden"], mc["kernel_layers"], mc["encoder_layers"], mc["decoder_layers"])
    elif model_type == "dsmpnn":
        base = S_MPNN(mc["node_in_channels"], mc["node_out_channels"], mc["edge_channels"],
                      mc["latent_dim"], mc["hops_h"], mc["encoder_hidden"], mc["decoder_hidden"],
                      mc["kernel_hidden"], mc["kernel_layers"], mc["encoder_layers"], mc["decoder_layers"])
        return DS_MPNN(base, use_communication=False)
    elif model_type == "gcn":
        return GCN(mc["node_in_channels"], mc["node_out_channels"], gc["hidden"], gc["layers"])
    raise ValueError(f"unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "smpnn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_config(cfg, model_type).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dc = cfg["data"]
    tc = cfg["training"]
    graphs = prepare_darcy_graphs(
        train_samples=dc["train_samples"], test_samples=dc["test_samples"],
        grid_size=dc["grid_size"], s=dc["sampled_nodes_s"], radius=dc["radius_r"],
        ne=dc["max_edges_ne"], seed=tc["seed"], normalize=dc.get("normalize", True),
    )
    test_graphs = graphs["test"]

    total_rmse, total_l1, n = 0.0, 0.0, 0
    with torch.no_grad():
        for g in test_graphs:
            g = g.to(device)
            out = model(g)
            target = g.y
            mask = g.interior_mask
            total_rmse += rmse(out[mask], target[mask]).item()
            total_l1 += F.l1_loss(out[mask], target[mask]).item()
            n += 1

    avg_rmse = total_rmse / max(n, 1)
    avg_l1 = total_l1 / max(n, 1)
    os.makedirs(args.output_dir, exist_ok=True)
    result = {"model_type": model_type, "test_rmse": avg_rmse, "test_l1": avg_l1, "n_test": n}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[save] {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
