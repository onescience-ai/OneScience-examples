"""Show the saved SFNO training and inference summaries."""

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
    rmse_by_step = torch.sqrt(error.square().mean(dim=(1, 2, 3)))
    pred_anom = prediction - prediction.mean(dim=(2, 3), keepdim=True)
    target_anom = target - target.mean(dim=(2, 3), keepdim=True)
    acc_by_step = (pred_anom * target_anom).sum(dim=(1, 2, 3)) / torch.sqrt(
        pred_anom.square().sum(dim=(1, 2, 3))
        * target_anom.square().sum(dim=(1, 2, 3))
    ).clamp_min(1e-12)
    metrics = {
        "rmse": float(torch.sqrt(error.square().mean())),
        "rmse_by_step": rmse_by_step.tolist(),
        "spatial_acc": float(acc_by_step.mean()),
        "spatial_acc_by_step": acc_by_step.tolist(),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, data, title in zip(
        axes,
        (target[0, 0], prediction[0, 0], error[0, 0]),
        ("Target", "Prediction", "Error"),
    ):
        image = ax.imshow(data, cmap="RdBu_r")
        ax.set_title(title)
        plt.colorbar(image, ax=ax, shrink=0.75)
    plt.tight_layout()
    plt.savefig(result_dir / "comparison.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))
