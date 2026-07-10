from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"


def main() -> int:
    score_path = RESULT_DIR / "score.json"
    pred_path = RESULT_DIR / "predictions.npz"
    if score_path.exists():
        print(score_path.read_text(encoding="utf-8"))
        return 0
    if not pred_path.exists():
        raise FileNotFoundError(f"No result found at {score_path} or {pred_path}. Run scripts/inference.py first.")
    data = np.load(pred_path)
    mse = float(np.mean((data["pred"] - data["target"]) ** 2))
    summary = {"mse": mse, "num_nodes": int(data["pred"].shape[0])}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
