import math
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from onescience.utils.YParams import YParams


def set_default_data_env():
    os.environ.setdefault("ONESCIENCE_BENO_DATA_DIR", str(PROJECT_ROOT / "data"))


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config():
    set_default_data_env()
    cfg = YParams(str(PROJECT_ROOT / "conf" / "config.yaml"), "root")
    cfg.datapipe.source.data_dir = str(resolve_path(cfg.datapipe.source.data_dir))
    cfg.datapipe.source.cache_dir = str(resolve_path(cfg.datapipe.source.cache_dir))
    cfg.fake_data.root_dir = str(resolve_path(cfg.fake_data.root_dir))
    return cfg


def square_boundary(points, boundary_points):
    bottom = [(x, 0.0) for x in points]
    top = [(x, 1.0) for x in points]
    left = [(0.0, y) for y in points[1:-1]]
    right = [(1.0, y) for y in points[1:-1]]
    boundary = np.array(bottom + top + left + right, dtype=np.float64)
    if boundary.shape[0] == 0:
        raise ValueError("resolution is too small to build boundary points")
    repeats = int(math.ceil(boundary_points / boundary.shape[0]))
    return np.tile(boundary, (repeats, 1))[:boundary_points]


def make_beno_arrays(cfg):
    data_cfg = cfg.datapipe.data
    fake_cfg = cfg.fake_data
    total_samples = int(data_cfg.ntrain) + int(data_cfg.ntest)
    resolution = int(data_cfg.resolution)
    boundary_points = int(fake_cfg.boundary_points)
    rng = np.random.default_rng(int(fake_cfg.seed))

    axis = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)

    cell_state = np.zeros((resolution, resolution), dtype=np.float64)
    cell_state[0, :] = 1.0
    cell_state[-1, :] = 1.0
    cell_state[:, 0] = 1.0
    cell_state[:, -1] = 1.0
    cell_state = cell_state.reshape(-1)

    boundary_xy = square_boundary(axis, boundary_points)

    rhs = np.zeros((total_samples, resolution * resolution, 4), dtype=np.float64)
    sol = np.zeros((total_samples, resolution * resolution, 1), dtype=np.float64)
    bc = np.zeros((total_samples, boundary_points, 4), dtype=np.float64)

    for sample_idx in range(total_samples):
        phase = 0.25 * sample_idx
        amplitude = 1.0 + 0.1 * sample_idx
        forcing = (
            amplitude * np.sin(np.pi * xx + phase) * np.sin(np.pi * yy)
            + 0.15 * np.cos(2.0 * np.pi * xx) * np.sin(np.pi * yy + phase)
        )
        solution = 0.25 * forcing + 0.05 * (xx + yy)
        if fake_cfg.noise_std:
            solution += float(fake_cfg.noise_std) * rng.standard_normal(solution.shape)

        boundary_value = (
            0.05 * sample_idx
            + 0.1 * boundary_xy[:, 0]
            - 0.08 * boundary_xy[:, 1]
        )

        rhs[sample_idx, :, 0:2] = coords
        rhs[sample_idx, :, 2] = forcing.reshape(-1)
        rhs[sample_idx, :, 3] = cell_state
        sol[sample_idx, :, 0] = solution.reshape(-1)
        bc[sample_idx, :, 0:2] = boundary_xy
        bc[sample_idx, :, 2] = boundary_value

    return rhs, sol, bc


def clear_matching_cache(cfg):
    cache_dir = Path(cfg.datapipe.source.cache_dir)
    prefix = cfg.datapipe.source.file_prefix
    for split, count in (("train", cfg.datapipe.data.ntrain), ("test", cfg.datapipe.data.ntest)):
        cache_file = cache_dir / f"cached_{prefix}_{split}_{count}.pt"
        if cache_file.exists():
            cache_file.unlink()


def main():
    cfg = load_config()
    data_dir = Path(cfg.datapipe.source.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    rhs, sol, bc = make_beno_arrays(cfg)
    prefix = cfg.datapipe.source.file_prefix
    np.save(data_dir / f"RHS_{prefix}_all.npy", rhs)
    np.save(data_dir / f"SOL_{prefix}_all.npy", sol)
    np.save(data_dir / f"BC_{prefix}_all.npy", bc)
    clear_matching_cache(cfg)

    print(f"Fake BENO data written to {data_dir}")
    print(f"RHS shape={rhs.shape}, SOL shape={sol.shape}, BC shape={bc.shape}")


if __name__ == "__main__":
    main()
