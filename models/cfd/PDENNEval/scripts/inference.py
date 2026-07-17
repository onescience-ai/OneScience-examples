from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Run PDENNEval FNO inference.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to conf/config.yaml")
    parser.add_argument("--data-dir", default=None, help="Override datapipe.source.data_dir")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Directory for npz predictions")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force-local-datapipe", action="store_true")
    args = parser.parse_args()

    dist = initialize_distributed()
    device = dist.device
    cfg = prepare_config(load_config(args.config), data_dir=args.data_dir, checkpoint=args.checkpoint)
    if args.output_dir:
        cfg.inference.output_dir = str(Path(args.output_dir).expanduser().resolve())

    checkpoint = Path(args.checkpoint or cfg.inference.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    datapipe = build_datapipe(
        cfg,
        distributed=False,
        force_local=args.force_local_datapipe,
    )
    val_loader, _ = datapipe.val_dataloader()
    model = build_model(datapipe.spatial_dim, cfg).to(device)
    model.load_state_dict(load_model_state(checkpoint, device))
    model.eval()

    output_dir = Path(args.output_dir or cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_batches = args.max_batches or int(get_attr(cfg.inference, "max_batches", 1))

    written = 0
    for batch_idx, (x, y, grid) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        grid = grid.to(device)
        pred, target = predict_batch(model, x, y, grid, cfg)
        path = output_dir / f"batch_{batch_idx:04d}.npz"
        np.savez_compressed(
            path,
            prediction=pred.detach().cpu().numpy(),
            target=target.detach().cpu().numpy(),
        )
        print(f"wrote {path}")
        written += 1

    if written == 0:
        raise RuntimeError("no validation batches were available for inference")

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
