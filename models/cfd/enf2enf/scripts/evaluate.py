"""评估入口：计算 Mean L2 Relative Error 并输出指标 JSON。

用法：
  python evaluate.py --config configs/elasticity.yaml --max-samples 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.elastic_dataset import ElasticityDataset  # noqa: E402
from metrics.l2_relative_error import l2_relative_error  # noqa: E402
from utils.utils import load_config, set_seed, load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    train_ds, test_ds = ElasticityDataset.build_splits(
        data_cfg["data_dir"], data_cfg["split_ratio"], data_cfg["seed"], None
    )
    ds = train_ds if args.split == "train" else test_ds
    bs = args.batch_size if args.batch_size else train_cfg["batch_size"]
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from train import build_model

    model = build_model(cfg).to(device)
    model.eval()

    enc_path = os.path.join(train_cfg["output_dir"], train_cfg["checkpoint_encoder"])
    dec_path = os.path.join(train_cfg["output_dir"], train_cfg["checkpoint_decoder"])
    if os.path.exists(enc_path):
        model.load_state_dict(load_checkpoint(enc_path), strict=False)
    if os.path.exists(dec_path):
        model.load_state_dict(load_checkpoint(dec_path), strict=False)

    rels = []
    with torch.no_grad():
        for coords, geom, sigma in loader:
            coords_d = coords.permute(0, 2, 1).to(device)
            geom_d = geom.permute(0, 2, 1).to(device)
            c = model.forward_encoder(coords_d, geom_d)
            u_hat = model.decode(coords_d, c, None).cpu()
            pred = ds.inverse_sigma(u_hat.reshape(coords.shape[0], -1))
            tgt = ds.inverse_sigma(sigma)  # raw units, per paper Mean L2 Relative Error
            rels.append(l2_relative_error(pred, tgt))
            if args.max_samples and len(rels) * bs >= args.max_samples:
                break

    metric = float(np.mean(rels))
    result = {
        "metric": "mean_l2_relative_error",
        "value": metric,
        "split": args.split,
        "n_samples": min(args.max_samples, len(ds)) if args.max_samples else len(ds),
        "paper_reference": 1.88e-2,
    }
    out_dir = args.out or train_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, cfg["eval"]["output_file"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"metric saved -> {out_path}")


if __name__ == "__main__":
    main()
