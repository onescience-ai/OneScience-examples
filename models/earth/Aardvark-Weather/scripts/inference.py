#!/usr/bin/env python
"""Run the one-day tas forecast through the official Aardvark modules."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import copy
import torch

from model.aardvark_adapter import build_one_day_model, load_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint produced by scripts/train.py")
    parser.add_argument("--output", type=Path, default=ROOT / "result" / "inference_one_day.json")
    args = parser.parse_args()
    root = args.root.resolve()
    sample_path = root / "weights/sample_data/sample_data_final.pkl"
    sample = load_sample(sample_path)
    model = build_one_day_model(root / "weights", root / "official-src", args.device)
    if args.checkpoint:
        tuned = args.checkpoint.resolve()
        payload = torch.load(tuned, map_location=args.device, weights_only=False)
        target_model = model if payload["train_modules"] == "all" else model.sf_model
        target_model.load_state_dict(payload["model"])
    model.eval()
    target = sample["y_target"].cpu()
    with torch.inference_mode():
        station, global_forecast, initial_state = model(copy.deepcopy(sample))
    result_dir = root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(station.cpu(), result_dir / "prediction.pt")
    torch.save(target, result_dir / "target.pt")
    report = {
        "device": args.device,
        "lead_time_days": 1,
        "station_tas_shape": list(station.shape),
        "global_forecast_shape": list(global_forecast.shape),
        "initial_state_shape": list(initial_state.shape),
        "finite_outputs": bool(torch.isfinite(station).all()),
        "weights": str(tuned) if args.checkpoint else "official",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
