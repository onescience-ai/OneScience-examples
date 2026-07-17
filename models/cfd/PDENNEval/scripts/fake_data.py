from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from common import DEFAULT_CONFIG, load_config, prepare_config


def generate_darcy_file(path: Path, samples: int, grid_size: int, seed: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Fake data already exists: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    nu = rng.uniform(0.05, 0.25, size=(samples, grid_size, grid_size)).astype(np.float32)
    tensor = np.empty((samples, 2, grid_size, grid_size), dtype=np.float32)
    base = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    for idx in range(samples):
        tensor[idx, 0] = base + 0.05 * rng.standard_normal(base.shape)
        tensor[idx, 1] = tensor[idx, 0] * (1.0 + nu[idx]) + 0.01 * rng.standard_normal(base.shape)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)
        handle.create_dataset("nu", data=nu)
        handle.create_dataset("x-coordinate", data=x)
        handle.create_dataset("y-coordinate", data=y)
        handle.create_dataset("t-coordinate", data=np.array([0.0, 1.0], dtype=np.float32))

    print(f"Fake Darcy HDF5 written: {path}")
    print(f"tensor={tensor.shape}, nu={nu.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a small PDENNEval FNO fake dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to conf/config.yaml")
    parser.add_argument("--data-dir", default=None, help="Override datapipe.source.data_dir")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = prepare_config(load_config(args.config), data_dir=args.data_dir)
    data_dir = Path(cfg.datapipe.source.data_dir)
    file_path = data_dir / cfg.datapipe.source.file_name
    generate_darcy_file(file_path, args.samples, args.grid_size, args.seed, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
