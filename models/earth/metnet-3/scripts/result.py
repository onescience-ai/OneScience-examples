"""Compute compact MetNet-3 fake-data metrics and visualizations."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.metnet3_heads import decode_bins


if __name__ == "__main__":
    result_dir = ROOT / "result"
    prediction = torch.load(result_dir / "prediction.pt", map_location="cpu", weights_only=True)
    target = torch.load(result_dir / "target.pt", map_location="cpu", weights_only=True)
    precipitation = decode_bins(prediction["precipitation_logits"])
    precipitation_target = target["precipitation"].float() / (prediction["precipitation_logits"].shape[2] - 1)
    ground = decode_bins(prediction["ground_logits"])
    ground_target = target["ground"].float() / (prediction["ground_logits"].shape[2] - 1)
    metrics = {
        "precipitation_mae_normalized": float((precipitation - precipitation_target).abs().mean()),
        "ground_mae_normalized": float((ground - ground_target).abs().mean()),
        "hrrr_rmse_normalized": float(torch.sqrt((prediction["hrrr_regression"] - target["hrrr"]).square().mean())),
        "probability_sum_error": float((prediction["precipitation_logits"].softmax(2).sum(2) - 1).abs().max()),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, data, title in zip(
        axes,
        (precipitation[0, 0], precipitation_target[0, 0], (precipitation - precipitation_target)[0, 0]),
        ("Prediction", "Target", "Error"),
    ):
        image = ax.imshow(data, cmap="RdBu_r")
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(image, ax=ax, shrink=0.75)
    plt.tight_layout()
    plt.savefig(result_dir / "comparison.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))
