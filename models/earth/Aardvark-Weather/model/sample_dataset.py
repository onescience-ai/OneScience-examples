"""Datasets for training Aardvark from official-schema pickle tasks."""

from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .aardvark_adapter import validate_sample


def discover_samples(path: Path) -> list[Path]:
    path = path.resolve()
    samples = sorted(path.glob("*.pkl")) if path.is_dir() else [path]
    if not samples or any(not sample.is_file() for sample in samples):
        raise FileNotFoundError(f"No Aardvark sample pickle found at {path}")
    for sample in samples:
        validate_sample(sample)
    return samples


def split_samples(samples: list[Path], validation_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(samples) == 1:
        return samples, samples
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


class AardvarkTaskDataset(Dataset):
    """Repeat one or more already-batched official tasks for a fixed number of steps."""

    def __init__(self, samples: list[Path], steps: int) -> None:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        self.tasks = [self._load(path) for path in samples]
        self.steps = steps

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            return pickle.load(handle)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> dict[str, Any]:
        return copy.deepcopy(self.tasks[index % len(self.tasks)])


def collate_tasks(items: list[Any]) -> Any:
    """Concatenate the batch dimension already present in official task tensors."""
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(items, dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate(items, axis=0)
    if isinstance(first, dict):
        return {key: collate_tasks([item[key] for item in items]) for key in first}
    if isinstance(first, tuple):
        return tuple(collate_tasks(list(values)) for values in zip(*items))
    if isinstance(first, list):
        return [collate_tasks(list(values)) for values in zip(*items)]
    if all(item == first for item in items):
        return first
    raise TypeError(f"Cannot collate Aardvark values of type {type(first).__name__}")
