"""Run inference on a few AirfRANS cases and save predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from funcattn_airfrans.data.airfrans import AirfransFieldDataset, collate_field, load_airfrans_field_stats  # noqa: E402
from funcattn_airfrans.models import FunctionalAttentionRegressor  # noqa: E402
from funcattn_airfrans.utils.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output", default="weight/predictions_reynolds.npz")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_root = resolve_path(ROOT, data_cfg["root"])
    cache_dir = resolve_path(ROOT, data_cfg["cache_dir"])
    stats = load_airfrans_field_stats(resolve_path(ROOT, data_cfg["stats_path"])) if data_cfg.get("normalize", True) else None
    split = args.split or data_cfg["test_split"]
    checkpoint = resolve_path(ROOT, args.checkpoint or train_cfg["checkpoint"])

    ds = AirfransFieldDataset(
        data_root,
        split,
        max_points=int(data_cfg.get("eval_max_points", data_cfg["max_points"])),
        limit=int(args.num_samples),
        cache_dir=cache_dir,
        stats=stats,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_field)
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
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    mean_out = std_out = None
    if stats is not None:
        mean_out = torch.as_tensor(stats["mean_out"], dtype=torch.float32, device=device).view(1, 1, -1)
        std_out = torch.as_tensor(stats["std_out"], dtype=torch.float32, device=device).view(1, 1, -1)

    saved = {}
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["x"].to(device), batch["mask"].to(device))
            if mean_out is not None and std_out is not None:
                pred = pred * (std_out + 1e-8) + mean_out
            name = batch["name"][0]
            valid = batch["mask"][0].bool().numpy()
            saved[f"{name}_pred"] = pred[0].detach().cpu().numpy()[valid]
            saved[f"{name}_points"] = batch["points"][0].numpy()[valid]

    output = resolve_path(ROOT, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **saved)
    print(f"predictions={output}")


if __name__ == "__main__":
    main()
