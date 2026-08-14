"""工具：配置加载、随机种子、checkpoint。"""

from __future__ import annotations

import os

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")
