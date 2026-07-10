from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from onescience.utils.YParams import YParams


def main() -> None:
    cfg = YParams(str(ROOT / "conf/config.yaml"), "model")
    cfg_test = YParams(str(ROOT / "conf/config.yaml"), "inference")
    result_root = ROOT / cfg_test.result_dir / cfg.name
    npy_dir = result_root / "npy"
    pred_files = sorted(npy_dir.glob("*_pred.npy")) if npy_dir.exists() else []
    gt_files = sorted(npy_dir.glob("*_gt.npy")) if npy_dir.exists() else []

    print(f"Result directory: {result_root}")
    print(f"Prediction files: {len(pred_files)}")
    print(f"Ground-truth files: {len(gt_files)}")
    if pred_files:
        print(f"First prediction: {pred_files[0]}")


if __name__ == "__main__":
    main()
