from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from common import (
    DEFAULT_CONFIG,
    build_datapipe,
    build_model,
    cleanup_distributed,
    get_attr,
    initialize_distributed,
    load_config,
    load_model_state,
    predict_batch,
    prepare_config,
)


def train_one_epoch(model, loader, optimizer, loss_fn, device, cfg) -> float:
    model.train()
    losses = []
    for x, y, grid in loader:
        x = x.to(device)
        y = y.to(device)
        grid = grid.to(device)
        pred, target = predict_batch(model, x, y, grid, cfg)
        if pred.numel() == 0:
            continue
        loss = loss_fn(pred.reshape(pred.size(0), -1), target.reshape(target.size(0), -1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg) -> float:
    model.eval()
    losses = []
    for x, y, grid in loader:
        x = x.to(device)
        y = y.to(device)
        grid = grid.to(device)
        pred, target = predict_batch(model, x, y, grid, cfg)
        if pred.numel() == 0:
            continue
        loss = loss_fn(pred.reshape(pred.size(0), -1), target.reshape(target.size(0), -1))
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the standard PDENNEval FNO package.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to conf/config.yaml")
    parser.add_argument("--data-dir", default=None, help="Override datapipe.source.data_dir")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir")
    parser.add_argument("--force-local-datapipe", action="store_true")
    args = parser.parse_args()

    dist = initialize_distributed()
    device = dist.device
    cfg = prepare_config(load_config(args.config), data_dir=args.data_dir, output_dir=args.output_dir)

    seed = int(get_attr(cfg.training, "seed", 0))
    torch.manual_seed(seed + dist.rank)
    np.random.seed(seed + dist.rank)

    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        print(f"Device: {device}, world_size={dist.world_size}")

    datapipe = build_datapipe(
        cfg,
        distributed=(dist.world_size > 1),
        force_local=args.force_local_datapipe,
    )
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, _ = datapipe.val_dataloader()

    model = build_model(datapipe.spatial_dim, cfg).to(device)
    if dist.world_size > 1:
        model = DDP(model, device_ids=[dist.local_rank], output_device=dist.local_rank)

    optimizer_cfg = cfg.training.optimizer
    scheduler_cfg = cfg.training.scheduler
    optimizer = getattr(torch.optim, optimizer_cfg.name)(
        model.parameters(),
        lr=float(optimizer_cfg.lr),
        weight_decay=float(optimizer_cfg.weight_decay),
    )
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_cfg.name)(
        optimizer,
        step_size=int(scheduler_cfg.step_size),
        gamma=float(scheduler_cfg.gamma),
    )
    loss_fn = nn.MSELoss()

    if bool(get_attr(cfg.training, "continue_training", False)) and cfg.training.model_path:
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(load_model_state(Path(cfg.training.model_path), device))

    best_val = float("inf")
    epochs = int(cfg.training.epochs)
    save_period = max(1, int(cfg.training.save_period))

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg)
        scheduler.step()
        val_loss = validate(model, val_loader, loss_fn, device, cfg)

        if dist.rank == 0:
            print(f"epoch={epoch} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")
            model_to_save = model.module if hasattr(model, "module") else model
            state = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            }
            torch.save(state, output_dir / "latest_model.pt")
            if (epoch + 1) % save_period == 0:
                torch.save(state, output_dir / f"model_epoch_{epoch}.pt")
            if val_loss <= best_val:
                best_val = val_loss
                torch.save(state, output_dir / "best_model.pt")

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
