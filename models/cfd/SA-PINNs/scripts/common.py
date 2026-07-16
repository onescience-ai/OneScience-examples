from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "root" not in config:
        raise ValueError(f"config must contain a 'root' mapping: {path}")
    return config["root"]


def project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but torch.cuda.is_available() is false")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_state(checkpoint: Mapping) -> tuple[Mapping[str, torch.Tensor], dict]:
    if "model_state" in checkpoint:
        return checkpoint["model_state"], dict(checkpoint)
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, {}
    raise ValueError("checkpoint contains no valid SA-PINN model state")
