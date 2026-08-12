"""Load the compact MetNet-3 checkpoint and run fake-data inference."""

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import MetNet3, MetNet3Config
from model.fake_data import make_fake


def main() -> None:
    result_dir = ROOT / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(ROOT / "weight" / "model.pth", map_location="cpu", weights_only=True)
    config = MetNet3Config(**checkpoint["config"])
    model = MetNet3(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    batch, targets = make_fake(config)
    with torch.inference_mode():
        outputs = model(batch)
    torch.save(outputs, result_dir / "prediction.pt")
    torch.save(targets, result_dir / "target.pt")
    summary = {name: list(value.shape) for name, value in outputs.items()}
    summary["finite"] = all(bool(torch.isfinite(value).all()) for value in outputs.values())
    (result_dir / "inference.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
