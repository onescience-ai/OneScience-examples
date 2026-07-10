import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from onescience.utils.YParams import YParams


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_fields(num_samples, height, width, seed):
    rng = np.random.default_rng(seed)
    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")

    x = np.empty((num_samples, 3, height, width), dtype=np.float32)
    target = np.empty_like(x)

    for i in range(num_samples):
        cx = 0.2 * np.sin(i)
        cy = 0.2 * np.cos(i * 0.7)
        radius = 0.25 + 0.03 * (i % 3)
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius
        obstacle = (distance < 0.0).astype(np.float32)
        inlet = np.ones_like(xx, dtype=np.float32) * (1.0 + 0.05 * i)

        x[i, 0] = obstacle
        x[i, 1] = distance
        x[i, 2] = inlet

        ux = inlet * (1.0 - obstacle) + 0.1 * np.sin(np.pi * yy)
        uy = -0.2 * yy * np.exp(-4.0 * np.maximum(distance, 0.0))
        pressure = 0.5 * (1.0 - xx) + 0.1 * obstacle

        noise = 0.005 * rng.standard_normal((3, height, width)).astype(np.float32)
        target[i, 0] = ux
        target[i, 1] = uy
        target[i, 2] = pressure
        target[i] += noise

    return x, target


def main():
    cfg = YParams(str(PROJECT_ROOT / "config" / "config.yaml"), "root")
    data_dir = resolve_path(cfg.datapipe.source.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    x, y = make_fields(
        num_samples=cfg.fake_data.num_samples,
        height=cfg.fake_data.height,
        width=cfg.fake_data.width,
        seed=cfg.fake_data.seed,
    )

    with open(data_dir / cfg.datapipe.source.data_x_name, "wb") as f:
        pickle.dump(x, f)
    with open(data_dir / cfg.datapipe.source.data_y_name, "wb") as f:
        pickle.dump(y, f)

    print(f"Fake DeepCFD data written to {data_dir}")
    print(f"x shape={x.shape}, y shape={y.shape}, dtype={x.dtype}")


if __name__ == "__main__":
    main()
