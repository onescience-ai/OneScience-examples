"""enf2enf 分阶段训练入口（Table 2: encoder epochs 1500 / decoder epochs 2500）。

阶段 1：训练 encoder（L^a 输入几何重建，含 CAVIA 内循环）。
阶段 2：冻结 encoder，训练 decoder（L^u 输出物理场）。

用法：
  python train.py --config configs/elasticity.yaml --phase encoder|decoder|all
  python train.py --config configs/elasticity.yaml --phase all --epochs-encoder 5 --epochs-decoder 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.elastic_dataset import ElasticityDataset  # noqa: E402
from losses.mse_loss import geometry_reconstruction_loss, field_reconstruction_loss  # noqa: E402
from models.enf2enf import ENF2ENF  # noqa: E402
from utils.utils import load_config, set_seed, save_checkpoint, load_checkpoint  # noqa: E402


def build_model(cfg: dict):
    m = cfg["model"]
    return ENF2ENF(
        coord_dim=m.get("coord_dim", 2),
        n_lat=m["n_lat"],
        l_dim=m["l_dim"],
        width_enc=m["width_enc"],
        width_dec=m["width_dec"],
        heads=m["heads"],
        sigma_window=m["sigma_window"],
        rff_d_enc=m["rff_d_enc"],
        rff_d_dec=m["rff_d_dec"],
        rff_sigma_enc=m["rff_sigma_enc"],
        rff_sigma_dec=m["rff_sigma_dec"],
        latent_bbox=tuple(m["latent_bbox"]),
        attention_blocks=m["attention_blocks"],
        inner_steps_K=m["inner_steps_K"],
        inner_lr_lambda_c=m["inner_lr_lambda_c"],
        geom_out_channels=2,
        field_out_channels=1,
        train_rff=m["train_rff"],
        use_global_params=m["use_global_params"],
        global_param_dim=0,
        share_input_decoder=m["share_input_decoder"],
        seed=cfg["data"]["seed"],
    )


def run_epoch_encoder(model, loader, opt, device):
    model.train()
    total = 0.0
    n = 0
    for coords, geom, _ in loader:
        coords = coords.permute(0, 2, 1).to(device)  # (B,N,2)
        geom = geom.permute(0, 2, 1).to(device)  # (B,N,2)
        opt.zero_grad()
        # 编码（CAVIA 内循环）+ 输入解码重建
        c = model.forward_encoder(coords, geom)
        a_hat = model.reconstruct_geometry(coords, c)
        loss = geometry_reconstruction_loss(a_hat, geom)
        loss.backward()
        opt.step()
        total += loss.item() * coords.shape[0]
        n += coords.shape[0]
    return total / max(n, 1)


def run_epoch_decoder(model, loader, opt, device, detach_encoder=True):
    model.train()
    total = 0.0
    n = 0
    for coords, geom, sigma in loader:
        coords = coords.permute(0, 2, 1).to(device)
        geom = geom.permute(0, 2, 1).to(device)
        sigma = sigma.unsqueeze(-1).to(device)  # (B,N,1)
        opt.zero_grad()
        c = model.forward_encoder(coords, geom)
        if detach_encoder:
            c = c.detach()
        u_hat = model.decode(coords, c, None)  # (B,1,N)
        loss = field_reconstruction_loss(u_hat, sigma)
        loss.backward()
        opt.step()
        total += loss.item() * coords.shape[0]
        n += coords.shape[0]
    return total / max(n, 1)


def evaluate_l2(model, loader, device, dataset):
    model.eval()
    errs = []
    with torch.no_grad():
        for coords, geom, sigma in loader:
            coords = coords.permute(0, 2, 1).to(device)
            geom = geom.permute(0, 2, 1).to(device)
            sigma = sigma.unsqueeze(1).to(device)
            c = model.forward_encoder(coords, geom)
            u_hat = model.decode(coords, c, None)
            pred = dataset.inverse_sigma(u_hat.cpu()).reshape(u_hat.shape[0], -1)
            tgt = dataset.inverse_sigma(sigma.cpu()).reshape(sigma.shape[0], -1)
            num = torch.norm(pred - tgt, dim=1)
            den = torch.norm(tgt, dim=1) + 1e-12
            errs.append((num / den).mean().item())
    return sum(errs) / max(len(errs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", choices=["encoder", "decoder", "all"], default="all")
    ap.add_argument("--epochs-encoder", type=int, default=None)
    ap.add_argument("--epochs-decoder", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="Tier1 快速复现：仅用前 N 个样本")
    ap.add_argument("--max-points", type=int, default=None, help="每样本点数子采样")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    data_cfg = cfg["data"]
    data_dir = data_cfg["data_dir"]
    split_ratio = data_cfg["split_ratio"]
    seed = args.seed if args.seed is not None else data_cfg["seed"]
    max_points = args.max_points

    train_ds, test_ds = ElasticityDataset.build_splits(data_dir, split_ratio, seed, max_points)
    if args.max_samples is not None:
        # 从 train/test 取子集用于快速复现
        import numpy as np

        train_ds.xy = train_ds.xy[: args.max_samples]
        train_ds.sigma = train_ds.sigma[: args.max_samples]
        test_ds.xy = test_ds.xy[: max(1, args.max_samples // 4)]
        test_ds.sigma = test_ds.sigma[: max(1, args.max_samples // 4)]

    bs = args.batch_size if args.batch_size else cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    print(f"model on {device}, latent positions:\n{model.latent_pos}")

    train_cfg = cfg["train"]
    out_dir = train_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    e_epochs = args.epochs_encoder if args.epochs_encoder is not None else train_cfg["epochs_encoder"]
    d_epochs = args.epochs_decoder if args.epochs_decoder is not None else train_cfg["epochs_decoder"]
    lr = train_cfg["lr_lambda_theta"]

    enc_opt = torch.optim.Adam([p for p in model.encoder.parameters() if p.requires_grad], lr=lr)
    dec_opt = torch.optim.Adam(
        [p for p in model.decoder.parameters() if p.requires_grad]
        + ([p for p in model.input_decoder.parameters() if p.requires_grad] if not cfg["model"]["share_input_decoder"] else []),
        lr=lr,
    )

    log_path = os.path.join(out_dir, "train.log")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        logf.write(line + "\n")
        logf.flush()

    if args.phase in ("encoder", "all"):
        for ep in range(e_epochs):
            loss = run_epoch_encoder(model, train_loader, enc_opt, device)
            if (ep + 1) % 10 == 0 or ep == 0:
                l2 = evaluate_l2(model, test_loader, device, test_ds)
                log(f"[encoder] epoch {ep+1}/{e_epochs} L_a={loss:.6f} test_l2={l2:.4f}")
        save_checkpoint(os.path.join(out_dir, train_cfg["checkpoint_encoder"]), model.state_dict())
        log("encoder checkpoint saved")

    if args.phase in ("decoder", "all"):
        # 阶段 2 需要 encoder 权重（若刚训完则已在内存）
        for ep in range(d_epochs):
            loss = run_epoch_decoder(model, train_loader, dec_opt, device)
            if (ep + 1) % 10 == 0 or ep == 0:
                l2 = evaluate_l2(model, test_loader, device, test_ds)
                log(f"[decoder] epoch {ep+1}/{d_epochs} L_u={loss:.6f} test_l2={l2:.4f}")
        save_checkpoint(os.path.join(out_dir, train_cfg["checkpoint_decoder"]), model.state_dict())
        log("decoder checkpoint saved")

    final_l2 = evaluate_l2(model, test_loader, device, test_ds)
    log(f"FINAL test Mean L2 Relative Error = {final_l2:.4f}")
    logf.close()


if __name__ == "__main__":
    main()
