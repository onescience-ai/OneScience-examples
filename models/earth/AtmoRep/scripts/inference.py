#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.tiny_atmorep import TinyAtmoRep, TinyAtmoRepConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weight" / "tiny_atmorep.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "result" / "prediction.pt")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TinyAtmoRepConfig(**payload["config"])
    model = TinyAtmoRep(config)
    model.load_state_dict(payload["model"])
    model.eval()
    torch.manual_seed(args.seed)
    fields = torch.randn(1, *config.input_shape)
    mask = torch.zeros(1, model.num_tokens, dtype=torch.bool)
    mask[:, 1::4] = True
    with torch.inference_mode():
        ensemble = model(fields, mask, level=137.0)
    target = model.tokenize(fields)
    result = {
        "ensemble": ensemble,
        "ensemble_mean": ensemble.mean(dim=1),
        "ensemble_std": ensemble.std(dim=1, unbiased=False),
        "mask": mask,
        "target": target,
        "input_shape": tuple(fields.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target, args.output.parent / "target.pt")
    torch.save(result, args.output)
    print(json.dumps({
        "output": str(args.output),
        "ensemble_shape": list(ensemble.shape),
        "mean_shape": list(result["ensemble_mean"].shape),
        "finite": bool(torch.isfinite(ensemble).all()),
        "bytes": args.output.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
