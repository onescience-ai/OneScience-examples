"""Load the trained DLWP-CS smoke checkpoint and run two-step inference."""

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import DLWPCubeSphereUNet, make_fake_batch, rollout


def main() -> None:
    result_dir = ROOT / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / "weight" / "model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint.get("model_config", {"in_channels": 2, "out_channels": 2, "base_channels": 4})
    model = DLWPCubeSphereUNet(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    inputs = make_fake_batch()
    with torch.inference_mode():
        prediction = rollout(model, inputs, 2)
    target = torch.stack((make_fake_batch(seed=8), make_fake_batch(seed=9)), dim=1)
    torch.save(prediction, result_dir / "prediction.pt")
    torch.save(target, result_dir / "target.pt")
    summary = {
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "finite": bool(torch.isfinite(prediction).all()),
    }
    (result_dir / "inference.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
