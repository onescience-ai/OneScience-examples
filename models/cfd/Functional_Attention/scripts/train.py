"""Train Functional Attention on the original AirfRANS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from funcattn_airfrans.data.airfrans import (  # noqa: E402
    AirfransFieldDataset,
    collate_field,
    compute_airfrans_field_stats,
    load_airfrans_field_stats,
)
from funcattn_airfrans.models import FunctionalAttentionRegressor  # noqa: E402
from funcattn_airfrans.modules import airfrans_loss  # noqa: E402
from funcattn_airfrans.utils.config import load_config, resolve_path  # noqa: E402


def _stats(cfg: dict, data_root: Path, cache_dir: Path, stats_path: Path) -> dict | None:
    if not bool(cfg["data"].get("normalize", True)):
        return None
    if stats_path.exists():
        print(f"loading_stats={stats_path}", flush=True)
        return load_airfrans_field_stats(stats_path)
    print(f"computing_stats={stats_path}", flush=True)
    return compute_airfrans_field_stats(
        data_root,
        cfg["data"]["train_split"],
        stats_path=stats_path,
        cache_dir=cache_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    torch.manual_seed(int(train_cfg.get("seed", 42)))

    data_root = resolve_path(ROOT, data_cfg["root"])
    cache_dir = resolve_path(ROOT, data_cfg["cache_dir"])
    stats_path = resolve_path(ROOT, data_cfg["stats_path"])
    checkpoint = resolve_path(ROOT, train_cfg["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    stats = _stats(cfg, data_root, cache_dir, stats_path)
    train_ds = AirfransFieldDataset(
        data_root,
        data_cfg["train_split"],
        max_points=int(data_cfg["max_points"]),
        cache_dir=cache_dir,
        stats=stats,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_field,
    )

    model = FunctionalAttentionRegressor(
        input_dim=int(model_cfg["input_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        channels=int(model_cfg["channels"]),
        layers=int(model_cfg["layers"]),
        heads=int(model_cfg["heads"]),
        bases=int(model_cfg["bases"]),
        ffn_ratio=int(model_cfg.get("ffn_ratio", 4)),
        ridge_lambda=float(model_cfg.get("ridge_lambda", 1e-3)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        share_basis=bool(model_cfg.get("share_basis", True)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    opt_cls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=float(train_cfg["lr"]))

    start_epoch = 0
    if bool(train_cfg.get("resume", True)) and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))
        print(f"resumed_checkpoint={checkpoint} start_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch + 1, int(train_cfg["epochs"]) + 1):
        model.train()
        total = volume = surface = 0.0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            surf = batch["surf"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss, lv, ls = airfrans_loss(
                pred,
                y,
                mask,
                surf,
                surface_weight=float(train_cfg.get("surface_loss_weight", 1.0)),
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            volume += float(lv.detach().cpu())
            surface += float(ls.detach().cpu())

        denom = max(len(loader), 1)
        last_loss = total / denom
        print(
            f"epoch={epoch} train_lv_plus_ls={last_loss:.6f} "
            f"train_lv={volume / denom:.6f} train_ls={surface / denom:.6f}",
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "last_loss": last_loss,
            },
            checkpoint,
        )

    print(f"checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
