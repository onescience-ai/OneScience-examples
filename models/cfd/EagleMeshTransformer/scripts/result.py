import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    output_dir = PROJECT_ROOT / "result" / "output"
    files = sorted(output_dir.glob("prediction_*.npy"))
    if not files:
        print(f"No prediction files found in {output_dir}")
        return

    print(f"Found {len(files)} prediction file(s) in {output_dir}")
    for file in files[:5]:
        arr = np.load(file)
        print(f"{file.name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")


if __name__ == "__main__":
    main()
