import os
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    os.chdir(PROJECT_ROOT)
    cfg = load_config()
    data_dir = PROJECT_ROOT / cfg["paths"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    n_total = cfg["data"]["ntrain"] + cfg["data"]["ntest"] + 1
    height, width = 221, 51
    rng = np.random.default_rng(42)

    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_axis = np.linspace(0.0, 1.0, width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y_axis, x_axis, indexing="ij")

    input_x = np.broadcast_to(grid_x, (n_total, height, width)).copy()
    input_y = np.broadcast_to(grid_y, (n_total, height, width)).copy()
    q = np.zeros((n_total, 5, height, width), dtype=np.float32)

    for i in range(n_total):
        phase = 0.15 * i
        field = np.sin(np.pi * (grid_x + phase)) * np.cos(np.pi * grid_y)
        q[i, 4] = field + 0.01 * rng.standard_normal((height, width))

    np.save(data_dir / "NACA_Cylinder_X.npy", input_x.astype(np.float32))
    np.save(data_dir / "NACA_Cylinder_Y.npy", input_y.astype(np.float32))
    np.save(data_dir / "NACA_Cylinder_Q.npy", q.astype(np.float32))

    print(f"Fake airfoil data written to {data_dir}")
    print(f"Samples={n_total}, X/Y shape={input_x.shape}, Q shape={q.shape}")


if __name__ == "__main__":
    main()
