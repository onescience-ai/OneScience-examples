"""Darcy 数据集加载器：将生成的原始场转换为 PyG Data 图列表。"""

from __future__ import annotations

import json
import os
import torch
import numpy as np

from dsmpnn.data.darcy import build_darcy_dataset
from dsmpnn.data.graph_builder import build_graphs


def prepare_darcy_graphs(train_samples: int, test_samples: int, grid_size: int,
                         s: int, radius: float, ne: int,
                         seed: int = 42, normalize: bool = True, nproc: int = 1,
                         cache_dir: str | None = None) -> dict:
    """构建 Darcy 训练/测试图。支持缓存（cache_dir 存在则加载）。"""
    import os
    if cache_dir and os.path.exists(os.path.join(cache_dir, "test")):
        train_graphs = load_graphs(os.path.join(cache_dir, "train"))
        test_graphs = load_graphs(os.path.join(cache_dir, "test"))
        if len(train_graphs) != train_samples or len(test_graphs) != test_samples:
            print(f"[data] cache sample count mismatch (train {len(train_graphs)}!={train_samples}, "
                  f"test {len(test_graphs)}!={test_samples}), regenerating")
        else:
            with open(os.path.join(cache_dir, "stats.json")) as f:
                stats = json.load(f)
            return {"train": train_graphs, "test": test_graphs, "stats": stats}

    train_raw, test_raw = build_darcy_dataset(train_samples, test_samples, grid_size, seed=seed, nproc=nproc)
    # train_raw/test_raw 每个元素 (node_attr, target, coords)
    train_attrs = [torch.from_numpy(a).float() for a, _, _ in train_raw]
    train_targets = [torch.from_numpy(t).float() for _, t, _ in train_raw]
    train_coords = [torch.from_numpy(c).float() for _, _, c in train_raw]

    test_attrs = [torch.from_numpy(a).float() for a, _, _ in test_raw]
    test_targets = [torch.from_numpy(t).float() for _, t, _ in test_raw]
    test_coords = [torch.from_numpy(c).float() for _, _, c in test_raw]

    if normalize:
        # per-variable z-score：统计训练集 a 通道（通道 2）均值/方差
        all_a = torch.cat([a[:, 2:3] for a in train_attrs], dim=0)
        a_mean, a_std = all_a.mean(), all_a.std().clamp(min=1e-6)
        stats = {"a_mean": float(a_mean), "a_std": float(a_std)}
        for i in range(len(train_attrs)):
            train_attrs[i] = train_attrs[i].clone()
            train_attrs[i][:, 2:3] = (train_attrs[i][:, 2:3] - a_mean) / a_std
        for i in range(len(test_attrs)):
            test_attrs[i] = test_attrs[i].clone()
            test_attrs[i][:, 2:3] = (test_attrs[i][:, 2:3] - a_mean) / a_std
    else:
        stats = {"a_mean": 0.0, "a_std": 1.0}

    train_seeds = list(range(train_samples))
    test_seeds = list(range(train_samples, train_samples + test_samples))

    train_graphs = build_graphs(train_attrs, train_targets, train_coords, s, radius, ne, train_seeds)
    test_graphs = build_graphs(test_attrs, test_targets, test_coords, s, radius, ne, test_seeds)

    if cache_dir:
        save_graphs(train_graphs, os.path.join(cache_dir, "train"))
        save_graphs(test_graphs, os.path.join(cache_dir, "test"))
        with open(os.path.join(cache_dir, "stats.json"), "w") as f:
            json.dump(stats, f)

    return {"train": train_graphs, "test": test_graphs, "stats": stats}


def save_graphs(graphs, path: str):
    import torch_geometric.data as pygdata
    os.makedirs(path, exist_ok=True)
    for i, g in enumerate(graphs):
        torch.save(g, os.path.join(path, f"sample_{i}.pt"))


def load_graphs(path: str):
    files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    return [torch.load(os.path.join(path, f), weights_only=False) for f in files]
