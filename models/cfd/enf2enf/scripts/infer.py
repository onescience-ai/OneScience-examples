"""推理入口：加载权重 → 测试集坐标查询解码 → 反归一化保存。

用法：
  python infer.py --config configs/elasticity.yaml --split test --max-samples 40
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.elastic_dataset import ElasticityDataset  # noqa: E402
from models.enf2enf import ENF2ENF  # noqa: E402
from utils.utils import load_config, set_seed, load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None, help="输出目录，默认 train.output_dir/predictions")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    # 归一化统计量需来自 train；通过构建 train 获取 norm
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
        state = load_checkpoint(enc_path)
        model.load_state_dict(state, strict=False)
    if os.path.exists(dec_path):
        state = load_checkpoint(dec_path)
        model.load_state_dict(state, strict=False)

    out_dir = args.out or os.path.join(train_cfg["output_dir"], "predictions")
    os.makedirs(out_dir, exist_ok=True)

    preds, targets, coords_all = [], [], []
    with torch.no_grad():
        for coords, geom, sigma in loader:
            coords_d = coords.permute(0, 2, 1).to(device)
            geom_d = geom.permute(0, 2, 1).to(device)
            c = model.forward_encoder(coords_d, geom_d)
            u_hat = model.decode(coords_d, c, None).cpu()  # (B,N,1) normalized
            pred_norm = u_hat.reshape(coords.shape[0], -1)
            sigma_norm = sigma  # (B,N) normalized
            pred = ds.inverse_sigma(pred_norm)
            tgt = ds.inverse_sigma(sigma_norm)
            preds.append(pred.numpy())
            targets.append(tgt.numpy())
            coords_all.append(coords.numpy())
            if args.max_samples and len(np.concatenate(preds)) >= args.max_samples:
                break

    preds = np.concatenate(preds)[: args.max_samples] if args.max_samples else np.concatenate(preds)
    targets = np.concatenate(targets)[: args.max_samples] if args.max_samples else np.concatenate(targets)
    np.save(os.path.join(out_dir, "predictions.npy"), preds)
    np.save(os.path.join(out_dir, "targets.npy"), targets)
    print(f"saved predictions {preds.shape} -> {out_dir}/predictions.npy")
    print(f"saved targets    {targets.shape} -> {out_dir}/targets.npy")


if __name__ == "__main__":
    main()
