import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from onescience.utils.YParams import YParams
from model import infer_task_type


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_path(cfg):
    task_type = infer_task_type(cfg.model.name)
    output_dir = resolve_path(cfg.inference.output_dir)
    if cfg.inference.get("group_by_model", False):
        output_dir = output_dir / task_type / cfg.model.name
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize CFDBench inference outputs.")
    parser.add_argument("--model", default=None, help="Override root.model.name and read that model's inference outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = YParams(str(PROJECT_ROOT / "conf" / "config.yaml"), "root")
    if args.model:
        cfg.model.name = args.model
    elif os.environ.get("CFDBENCH_MODEL_NAME"):
        cfg.model.name = os.environ["CFDBENCH_MODEL_NAME"]
    output_dir = output_path(cfg)
    pred_path = output_dir / "preds.pt"
    score_path = output_dir / "scores.json"

    if not pred_path.is_file() or not score_path.is_file():
        raise FileNotFoundError("Missing inference outputs. Run scripts/inference.py first.")

    preds = torch.load(pred_path, map_location="cpu", weights_only=True)
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    print(f"Prediction tensor: shape={tuple(preds.shape)}, dtype={preds.dtype}")
    print(f"Prediction range: min={float(preds.min()):.4e}, max={float(preds.max()):.4e}")
    print(f"Scores: {scores['mean']}")


if __name__ == "__main__":
    main()
