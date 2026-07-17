import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from onescience.utils.YParams import YParams


def main():
    os.chdir(PROJECT_ROOT)
    cfg = YParams(PROJECT_ROOT / "config" / "config.yaml", "inference")
    result_path = PROJECT_ROOT / cfg.output_path
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}. Run scripts/inference.py first.")
    data = np.load(result_path)
    prediction = data["prediction"]
    target = data["target"]
    mse = float(np.mean((prediction - target) ** 2))
    print(f"result_path: {result_path.relative_to(PROJECT_ROOT)}")
    print(f"prediction_shape: {prediction.shape}")
    print(f"target_shape: {target.shape}")
    print(f"mse: {mse:.8f}")


if __name__ == "__main__":
    main()
