"""Show the saved Aardvark one-day inference summary."""

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
    valid = torch.isfinite(target)
    rmse = torch.sqrt(torch.mean((prediction[valid] - target[valid]) ** 2))
    mae = torch.mean((prediction[valid] - target[valid]).abs())
    metrics = {
        "normalized_rmse": float(rmse),
        "normalized_mae": float(mae),
        "valid_stations": int(valid.sum()),
        "note": "Normalized sample-space metrics; not paper-comparable physical-unit scores.",
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    indices = torch.where(valid[0])[0][:300]
    plt.figure(figsize=(10, 4))
    plt.plot(target[0, indices], label="Target", linewidth=1)
    plt.plot(prediction[0, indices], label="Prediction", linewidth=1)
    plt.title(f"Aardvark station temperature (RMSE={rmse:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "comparison.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))
