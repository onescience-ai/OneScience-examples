"""Configuration for the standalone SFNO smoke workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SFNOConfig:
    nlat: int = 17
    nlon: int = 32
    channels: int = 2
    timesteps: int = 6
    batch_size: int = 2
    embed_dim: int = 8
    num_layers: int = 2
    scale_factor: int = 2
    rollout_steps: int = 3
    learning_rate: float = 0.001
    seed: int = 2026
    grid: str = "equiangular"
    grid_internal: str = "legendre-gauss"

    def validate(self) -> None:
        positive = (
            "nlat",
            "nlon",
            "channels",
            "timesteps",
            "batch_size",
            "embed_dim",
            "num_layers",
            "scale_factor",
            "rollout_steps",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.nlat < 5 or self.nlon < 8:
            raise ValueError("The spherical grid must be at least 5 x 8")
        if self.nlon % self.scale_factor:
            raise ValueError("nlon must be divisible by scale_factor")
        if self.timesteps < self.batch_size + 1:
            raise ValueError("timesteps must provide at least batch_size input/target pairs")
        if self.rollout_steps >= self.timesteps:
            raise ValueError("rollout_steps must be smaller than timesteps")
        if self.grid not in {"equiangular", "legendre-gauss", "lobatto", "equidistant"}:
            raise ValueError(f"Unsupported input grid: {self.grid}")


def load_config(path: str | Path) -> SFNOConfig:
    path = Path(path)
    values: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    known = {field.name for field in fields(SFNOConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    config = SFNOConfig(**values)
    config.validate()
    return config
