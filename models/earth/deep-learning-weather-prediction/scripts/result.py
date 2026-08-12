"""Compute DLWP-CS fake-data metrics and visualization."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result_dir = ROOT / "result"
    prediction = torch.load(result_dir / "prediction.pt", map_location="cpu", weights_only=True)
    target = torch.load(result_dir / "target.pt", map_location="cpu", weights_only=True)
    error = prediction - target
    rmse_by_step = torch.sqrt(error.square().mean(dim=(0, 2, 3, 4, 5)))
    pred_anom = prediction - prediction.mean(dim=(3, 4, 5), keepdim=True)
    target_anom = target - target.mean(dim=(3, 4, 5), keepdim=True)
    corr = (pred_anom * target_anom).sum(dim=(3, 4, 5)) / torch.sqrt(
        pred_anom.square().sum(dim=(3, 4, 5))
        * target_anom.square().sum(dim=(3, 4, 5))
    ).clamp_min(1e-12)
    metrics = {
        "rmse": float(torch.sqrt(error.square().mean())),
        "rmse_by_step": rmse_by_step.tolist(),
        "spatial_acc": float(corr.mean()),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for face, ax in enumerate(axes.flat):
        image = ax.imshow(error[0, -1, 0, face], cmap="RdBu_r")
        ax.set_title(f"Face {face} error")
        ax.axis("off")
        plt.colorbar(image, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(result_dir / "comparison.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))
