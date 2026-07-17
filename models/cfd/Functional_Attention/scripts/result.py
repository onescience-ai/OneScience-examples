"""Evaluate a trained Functional Attention checkpoint."""

from __future__ import annotations

import argparse
import json
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
from funcattn_airfrans.utils.metrics import (  # noqa: E402
    airfrans_lv_ls_loss_torch,
    masked_relative_l2_torch,
    pressure_force_coefficients,
    spearmanr_numpy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_root = resolve_path(ROOT, data_cfg["root"])
    cache_dir = resolve_path(ROOT, data_cfg["cache_dir"])
    stats = load_airfrans_field_stats(resolve_path(ROOT, data_cfg["stats_path"])) if data_cfg.get("normalize", True) else None
    split = args.split or eval_cfg.get("split") or data_cfg["test_split"]
    checkpoint = resolve_path(ROOT, args.checkpoint or train_cfg["checkpoint"])

    ds = AirfransFieldDataset(
        data_root,
        split,
        max_points=int(data_cfg.get("eval_max_points", data_cfg["max_points"])),
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

    rel_l2 = []
    lv_values = []
    ls_values = []
    pred_lift = []
    true_lift = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            surf = batch["surf"].to(device)
            pred = model(x, mask)
            target = y
            if mean_out is not None and std_out is not None:
                pred = pred * (std_out + 1e-8) + mean_out
                target = target * (std_out + 1e-8) + mean_out
            rel_l2.append(float(masked_relative_l2_torch(pred, target, mask).detach().cpu()))
            _, lv, ls = airfrans_lv_ls_loss_torch(pred, target, mask, surf)
            lv_values.append(float(lv.detach().cpu()))
            ls_values.append(float(ls.detach().cpu()))
            valid_surf = (batch["surf"][0].bool() & batch["mask"][0].bool()).numpy()
            if np.any(valid_surf):
                pred_p = pred[0, :, 2].detach().cpu().numpy()[valid_surf]
                true_p = target[0, :, 2].detach().cpu().numpy()[valid_surf]
                normals = batch["normals"][0].numpy()[valid_surf]
                _, pred_cl = pressure_force_coefficients(pred_p, normals)
                _, true_cl = pressure_force_coefficients(true_p, normals)
                pred_lift.append(pred_cl)
                true_lift.append(true_cl)

    pred_lift_arr = np.asarray(pred_lift)
    true_lift_arr = np.asarray(true_lift)
    score = {
        "split": split,
        "relative_l2_mean": float(np.mean(rel_l2)),
        "volume_relative_l2_mean": float(np.mean(lv_values)),
        "surface_relative_l2_mean": float(np.mean(ls_values)),
        "pressure_only_force_metric": "approximate",
        "pressure_only_lift_relative_error": float(
            np.mean(np.abs(pred_lift_arr - true_lift_arr) / np.maximum(np.abs(true_lift_arr), 1e-12))
        ) if pred_lift else None,
        "spearman_lift": float(spearmanr_numpy(pred_lift_arr, true_lift_arr)) if pred_lift else None,
    }
    output = resolve_path(ROOT, eval_cfg.get("output", "weight/result.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(score, indent=2, ensure_ascii=False))
    print(f"result={output}")


if __name__ == "__main__":
    main()
