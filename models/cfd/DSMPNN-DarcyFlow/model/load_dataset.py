
import torch
from pathlib import Path

def load_graphs(dataset_path, split="train"):
    """加载 PyG Data 图列表。"""
    split_dir = Path(dataset_path) / split
    files = sorted(split_dir.glob("*.pt"))
    return [torch.load(f, weights_only=False) for f in files]

def load_stats(dataset_path):
    import json
    with open(Path(dataset_path) / "stats.json") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "."
    print("train:", len(load_graphs(ds, "train")))
    print("test:", len(load_graphs(ds, "test")))
    print("stats:", load_stats(ds))
