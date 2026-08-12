"""Load an SFNO smoke checkpoint and run autoregressive inference."""

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT))

from model.config import load_config
from model.fake_spherical_data import make_fake_spherical_sequence
from model.sfno_adapter import OfficialSFNOAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weight" / "model.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result")
    args = parser.parse_args()
    result_dir = args.output_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config_path = result_dir / "checkpoint_config.json"
    config_path.write_text(json.dumps(checkpoint["config"], indent=2) + "\n")
    config = load_config(config_path)
    model = OfficialSFNOAdapter(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    fields = make_fake_spherical_sequence(
        config.timesteps, config.channels, config.nlat, config.nlon, config.seed
    )["fields"]
    state = fields[0:1]
    target = fields[1 : config.rollout_steps + 1]
    forecasts = []
    with torch.inference_mode():
        for _ in range(config.rollout_steps):
            state = model(state)
            forecasts.append(state)
    prediction = torch.cat(forecasts, dim=0)
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
