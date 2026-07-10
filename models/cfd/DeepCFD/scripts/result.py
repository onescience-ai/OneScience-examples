import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from onescience.utils.YParams import YParams


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    cfg = YParams(str(PROJECT_ROOT / "config" / "config.yaml"), "root")
    checkpoint_path = resolve_path(cfg.inference.checkpoint_path)
    pred_dir = resolve_path(cfg.inference.result_dir) / "predictions"

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Epoch: {ckpt.get('epoch')}, val_loss: {ckpt.get('val_loss')}")
        print(f"Model config: {ckpt.get('config')}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

    pred_path = pred_dir / "prediction_batch.npy"
    if pred_path.exists():
        pred = np.load(pred_path)
        print(f"Prediction batch: shape={pred.shape}, dtype={pred.dtype}, mean={pred.mean():.6f}")
    else:
        print(f"Prediction batch not found: {pred_path}")


if __name__ == "__main__":
    main()
