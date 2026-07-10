import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))


def write_split(split_dir: Path, name: str, samples):
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / f"{name}.txt", "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(f"{sample}\n")


def make_sample(path: Path, seed: int, steps: int = 990, nodes: int = 6):
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)

    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.2],
            [0.5, 0.8],
        ],
        dtype=np.float32,
    )
    drift = np.linspace(0.0, 0.04, steps, dtype=np.float32).reshape(steps, 1, 1)
    pointcloud = base.reshape(1, nodes, 2) + drift

    t = np.arange(steps, dtype=np.float32).reshape(steps, 1)
    x = pointcloud[..., 0]
    y = pointcloud[..., 1]
    vx = np.sin(x + t * 0.1).astype(np.float32)
    vy = np.cos(y + t * 0.1).astype(np.float32)
    ps = (x + y + 0.01 * rng.standard_normal((steps, nodes))).astype(np.float32)
    pg = (x - y + 0.01 * rng.standard_normal((steps, nodes))).astype(np.float32)
    mask = np.zeros((steps, nodes), dtype=np.int64)

    triangles = np.array(
        [
            [0, 1, 4],
            [0, 4, 2],
            [1, 3, 4],
            [2, 4, 5],
            [3, 5, 4],
        ],
        dtype=np.int64,
    )
    triangles = np.repeat(triangles.reshape(1, -1, 3), steps, axis=0)

    np.savez(
        path / "sim.npz",
        pointcloud=pointcloud,
        VX=vx,
        VY=vy,
        PS=ps,
        PG=pg,
        mask=mask,
    )
    np.save(path / "triangles.npy", triangles)


def main():
    os.chdir(PROJECT_ROOT)
    data_root = PROJECT_ROOT / "data" / "fake" / "Eagle_dataset"
    split_dir = PROJECT_ROOT / "data" / "fake" / "splits"
    samples = ["Cre/fake_case_000/1", "Cre/fake_case_001/1", "Cre/fake_case_002/1"]

    for idx, sample in enumerate(samples):
        make_sample(data_root / sample, seed=idx)

    write_split(split_dir, "train", samples[:2])
    write_split(split_dir, "valid", samples[2:])
    write_split(split_dir, "test", samples[2:])
    print(f"Fake Eagle data written to: {data_root}")
    print(f"Fake splits written to: {split_dir}")


if __name__ == "__main__":
    main()
