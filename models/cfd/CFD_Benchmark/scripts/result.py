import json
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    checkpoint = PROJECT_ROOT / cfg["paths"]["weight_path"]
    result_dir = PROJECT_ROOT / cfg["paths"]["result_dir"] / cfg["train"]["save_name"]
    metrics_path = result_dir / "metrics.json"

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "result_dir": str(result_dir),
        "result_dir_exists": result_dir.exists(),
        "metrics": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
    }
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            summary["epoch"] = state.get("epoch")
            summary["best_epoch"] = state.get("best_epoch")
            summary["best_test_loss"] = state.get("best_test_loss")
            summary["has_model_state"] = "model_state" in state
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            summary["metrics_values"] = json.load(f)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
