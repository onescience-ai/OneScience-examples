from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from onescience.utils.YParams import YParams


def make_edges(num_nodes: int) -> np.ndarray:
    src = np.arange(num_nodes, dtype=np.int64)
    dst = (src + 1) % num_nodes
    edges = np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])],
        axis=0,
    )
    return edges


def write_placeholder_training_files(param_dir: Path, rng: np.random.Generator) -> None:
    param_dir.mkdir(parents=True, exist_ok=True)
    np.save(param_dir / "Cd.npy", rng.normal(size=(1,)).astype(np.float32))
    np.save(param_dir / "I1.npy", rng.normal(size=(8, 3)).astype(np.float32))
    np.save(param_dir / "I2.npy", rng.normal(size=(8, 3)).astype(np.float32))
    np.save(param_dir / "Press.npy", rng.normal(size=(8, 1)).astype(np.float32))
    np.save(param_dir / "Velo.npy", rng.normal(size=(8, 3)).astype(np.float32))


def write_sample(sample_dir: Path, num_nodes: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sample_dir.mkdir(parents=True, exist_ok=True)

    pos = rng.uniform(low=-1.0, high=1.0, size=(num_nodes, 3)).astype(np.float32)
    sdf = rng.uniform(low=0.0, high=0.2, size=(num_nodes, 1)).astype(np.float32)
    normals = rng.normal(size=(num_nodes, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6
    x = np.concatenate([pos, sdf, normals], axis=1).astype(np.float32)

    y = np.concatenate(
        [
            0.1 * pos + rng.normal(scale=0.01, size=(num_nodes, 3)),
            rng.normal(scale=0.05, size=(num_nodes, 1)),
        ],
        axis=1,
    ).astype(np.float32)

    surf = np.zeros((num_nodes,), dtype=np.bool_)
    surf[num_nodes // 2 :] = True
    edge_index = make_edges(num_nodes)

    np.save(sample_dir / "x.npy", x)
    np.save(sample_dir / "y.npy", y)
    np.save(sample_dir / "pos.npy", pos)
    np.save(sample_dir / "surf.npy", surf)
    np.save(sample_dir / "edge_index.npy", edge_index)
    return x, y


def main() -> None:
    cfg = YParams(str(ROOT / "conf/config.yaml"), "datapipe")
    data_dir = ROOT / cfg.source.data_dir
    preprocessed_dir = ROOT / cfg.source.preprocessed_save_dir
    stats_dir = ROOT / cfg.source.stats_dir
    stats_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for fold_id in range(9):
        param_name = f"param{fold_id}"
        param_dir = data_dir / param_name
        write_placeholder_training_files(param_dir, rng)

        sample_name = "sample_000"
        (param_dir / sample_name).mkdir(parents=True, exist_ok=True)
        x, y = write_sample(preprocessed_dir / param_name / sample_name, num_nodes=8, rng=rng)
        all_x.append(x)
        all_y.append(y)

    x_all = np.concatenate(all_x, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    np.save(stats_dir / "mean_in.npy", x_all.mean(axis=0).astype(np.float32))
    np.save(stats_dir / "std_in.npy", (x_all.std(axis=0) + 1e-6).astype(np.float32))
    np.save(stats_dir / "mean_out.npy", y_all.mean(axis=0).astype(np.float32))
    np.save(stats_dir / "std_out.npy", (y_all.std(axis=0) + 1e-6).astype(np.float32))

    print(f"Fake ShapeNetCar data generated under {ROOT / 'data/mlcfd_data'}")


if __name__ == "__main__":
    main()
