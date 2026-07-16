from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class AttrDict(dict):
    """Dictionary with recursive attribute access for OneScience datapipes."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def to_attr_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return AttrDict({key: to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_attr_dict(item) for item in value]
    return value


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("root"), dict):
        raise ValueError(f"Config must contain a root mapping: {path}")
    return config["root"]


def project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout(
    model: torch.nn.Module,
    pos: torch.Tensor,
    state: torch.Tensor,
    t_out: int,
    out_dim: int,
    max_latent_steps: int,
) -> torch.Tensor:
    predictions = []
    remaining = int(t_out)
    while remaining > 0:
        step_count = min(int(max_latent_steps), remaining)
        prediction = model(pos, state, latent_steps=step_count)
        predictions.append(prediction)
        consumed = step_count * int(out_dim)
        state = torch.cat((state[..., consumed:], prediction), dim=-1)
        remaining -= step_count
    return torch.cat(predictions, dim=-1)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(prediction - target, dim=1)
    denominator = torch.linalg.vector_norm(target, dim=1).clamp_min(1.0e-8)
    return numerator / denominator
