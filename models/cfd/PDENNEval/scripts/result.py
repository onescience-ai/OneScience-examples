from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import DEFAULT_CONFIG, load_config, prepare_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PDENNEval FNO inference outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to conf/config.yaml")
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = prepare_config(load_config(args.config))
    prediction_dir = Path(args.prediction_dir or cfg.result.prediction_dir).expanduser().resolve()
    output_dir = Path(args.output_dir or cfg.result.output_dir).expanduser().resolve()
    files = sorted(prediction_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no prediction files found in {prediction_dir}")

    mse_values = []
    mae_values = []
    for path in files:
        data = np.load(path)
        pred = data["prediction"]
        target = data["target"]
        diff = pred - target
        mse_values.append(float(np.mean(diff ** 2)))
        mae_values.append(float(np.mean(np.abs(diff))))

    metrics = {
        "num_files": len(files),
        "mse": float(np.mean(mse_values)),
        "mae": float(np.mean(mae_values)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
