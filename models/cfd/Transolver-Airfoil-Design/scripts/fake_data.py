from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from onescience.utils.YParams import YParams


def resolve_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def make_sample(num_nodes: int, input_dim: int, output_dim: int, surface_ratio: float, seed: int) -> Data:
    generator = torch.Generator().manual_seed(seed)
    pos = torch.rand((num_nodes, 2), generator=generator)
    freestream = torch.ones((num_nodes, 2))
    freestream[:, 1] = torch.linspace(-0.2, 0.2, num_nodes)
    sdf = torch.linalg.norm(pos - 0.5, dim=1, keepdim=True) - 0.25
    normals = torch.nn.functional.normalize(pos - 0.5, dim=1)
    x = torch.cat([pos, freestream, sdf, normals], dim=1)
    if x.size(1) != input_dim:
        raise ValueError(f"fake input_dim must be 7 for this AirfRANS feature layout, got {input_dim}")

    y0 = torch.sin(np.pi * pos[:, :1]) + 0.1 * freestream[:, :1]
    y1 = torch.cos(np.pi * pos[:, 1:2]) + 0.1 * freestream[:, 1:2]
    pressure = pos[:, :1] - pos[:, 1:2]
    viscosity = torch.abs(sdf) + 0.01
    y = torch.cat([y0, y1, pressure, viscosity], dim=1)
    if y.size(1) != output_dim:
        raise ValueError(f"fake output_dim must be 4 for this AirfRANS target layout, got {output_dim}")

    surf = torch.zeros(num_nodes, dtype=torch.bool)
    surf_count = max(1, int(num_nodes * surface_ratio))
    surf[:surf_count] = True
    perm = torch.randperm(num_nodes, generator=generator)
    return Data(pos=pos[perm], x=x[perm].float(), y=y[perm].float(), surf=surf[perm])


def main() -> int:
    cfg = YParams(str(ROOT / "conf" / "config.yaml"), "fake_data")
    cfg_data = YParams(str(ROOT / "conf" / "config.yaml"), "datapipe")
    cfg_train = YParams(str(ROOT / "conf" / "config.yaml"), "training")
    data_dir = resolve_path(cfg_data.source.data_dir)
    stats_dir = resolve_path(cfg_data.source.stats_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {
        cfg_data.data.splits.train_name: int(cfg.num_train),
        cfg_data.data.splits.val_name: int(cfg.num_val),
        cfg_data.data.splits.test_name: int(cfg.num_test),
    }
    manifest: dict[str, list[str]] = {}
    all_x, all_y = [], []
    seed = int(cfg_train.seed)

    for split, count in split_counts.items():
        names = []
        for idx in range(count):
            name = f"{split}_{idx:04d}"
            sample = make_sample(
                num_nodes=int(cfg.num_nodes),
                input_dim=int(cfg.input_dim),
                output_dim=int(cfg.output_dim),
                surface_ratio=float(cfg.surface_ratio),
                seed=seed + len(all_x),
            )
            torch.save(sample, data_dir / f"{name}.pt")
            names.append(name)
            if split == cfg_data.data.splits.train_name:
                all_x.append(sample.x.numpy())
                all_y.append(sample.y.numpy())
        manifest[split] = names

    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train_x = np.concatenate(all_x, axis=0)
    train_y = np.concatenate(all_y, axis=0)
    np.save(stats_dir / "mean_in.npy", train_x.mean(axis=0).astype(np.float32))
    np.save(stats_dir / "std_in.npy", (train_x.std(axis=0) + 1e-6).astype(np.float32))
    np.save(stats_dir / "mean_out.npy", train_y.mean(axis=0).astype(np.float32))
    np.save(stats_dir / "std_out.npy", (train_y.std(axis=0) + 1e-6).astype(np.float32))
    print(f"Fake AirfRANS graph data written to {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
