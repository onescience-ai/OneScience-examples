"""Show the saved tiny AtmoRep training and inference summaries."""

import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    history_path = ROOT / "weight" / "training" / "history.json"
    train_history = json.loads(history_path.read_text()) if history_path.exists() else []
    prediction = torch.load(ROOT / "result" / "prediction.pt", map_location="cpu", weights_only=True)
    target = torch.load(ROOT / "result" / "target.pt", map_location="cpu", weights_only=True)
    ensemble = prediction["ensemble"]
    target = target["target"] if isinstance(target, dict) else target
    mean = ensemble.mean(dim=1)
    std = ensemble.std(dim=1, unbiased=False)
    ensemble_rmse = torch.sqrt((ensemble - target[:, None]).square().mean())
    mean_rmse = torch.sqrt((mean - target).square().mean())
    target_std = target.std(dim=-1, unbiased=False, keepdim=True).expand_as(target)
    spread_rmse = torch.sqrt((std - target_std).square().mean())
    summary = {
        "train_history": train_history,
        "ensemble_shape": list(ensemble.shape),
        "target_shape": list(target.shape),
        "ensemble_rmse": float(ensemble_rmse),
        "mean_rmse": float(mean_rmse),
        "spread_rmse": float(spread_rmse),
        "finite": bool(torch.isfinite(ensemble).all()),
    }
    (ROOT / "result" / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, data, title in zip(axes, (target[0, 0], mean[0, 0], (mean - target)[0, 0].abs(), std[0, 0]), ("Target", "Mean", "Abs error", "Spread")):
        image = ax.imshow(data.reshape(4, 4).numpy(), cmap="viridis")
        ax.set_title(title)
        plt.colorbar(image, ax=ax, shrink=0.75)
    plt.tight_layout()
    plt.savefig(ROOT / "result" / "comparison.png", dpi=150)
    plt.close()
    print(json.dumps(summary, indent=2))
