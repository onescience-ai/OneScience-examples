"""Shared data, normalization, metric, and serialization utilities."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    for section in ("experiment", "paths", "data", "normalization", "model"):
        if section not in config:
            raise KeyError(f"missing required config section: {section}")
    return config


def project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def data_file(config: dict[str, Any], filename_key: str) -> Path:
    directory = Path(config["paths"]["data_dir"]).expanduser()
    path = (directory / config["paths"][filename_key]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"data file not found: {path}")
    return path


def numeric_sample_ids(split: dict[str, int]) -> list[int]:
    start, stop = int(split["start"]), int(split["stop"])
    if start < 0 or stop <= start:
        raise ValueError(f"invalid half-open sample range [{start}, {stop})")
    return list(range(start, stop))


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


@dataclass(frozen=True)
class MinMaxNormalizer:
    input_min: float
    input_max: float
    output_min: float
    output_max: float
    epsilon: float = 1.0e-12
    source: str = ""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MinMaxNormalizer":
        values = config["normalization"]
        result = cls(
            input_min=float(values["input_min"]),
            input_max=float(values["input_max"]),
            output_min=float(values["output_min"]),
            output_max=float(values["output_max"]),
            epsilon=float(values.get("epsilon", 1.0e-12)),
            source=str(values.get("source", "")),
        )
        result.validate()
        return result

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "MinMaxNormalizer":
        result = cls(**state)
        result.validate()
        return result

    def validate(self) -> None:
        values = (self.input_min, self.input_max, self.output_min, self.output_max)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"normalization contains nonfinite values: {values}")
        if self.input_max - self.input_min <= self.epsilon:
            raise ValueError("input normalization range is zero or negative")
        if self.output_max - self.output_min <= self.epsilon:
            raise ValueError("output normalization range is zero or negative")

    def normalize_input(self, value: Tensor) -> Tensor:
        return (value - self.input_min) / (self.input_max - self.input_min)

    def normalize_output(self, value: Tensor) -> Tensor:
        return (value - self.output_min) / (self.output_max - self.output_min)

    def denormalize_input(self, value: Tensor) -> Tensor:
        return value * (self.input_max - self.input_min) + self.input_min

    def denormalize_output(self, value: Tensor) -> Tensor:
        return value * (self.output_max - self.output_min) + self.output_min

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


class NavierStokesH5Dataset(Dataset[tuple[Tensor, Tensor, int]]):
    """Lazy reader for the supplied ``Sample_i/{input,output}`` benchmark."""

    def __init__(
        self,
        path: str | Path,
        sample_ids: Sequence[int],
        normalizer: MinMaxNormalizer,
        input_key: str = "input",
        output_key: str = "output",
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")
        self.sample_ids = [int(sample_id) for sample_id in sample_ids]
        if not self.sample_ids:
            raise ValueError("dataset sample_ids must not be empty")
        self.normalizer = normalizer
        self.input_key = input_key
        self.output_key = output_key
        self._handle: h5py.File | None = None
        self._validate_contract()

    def _validate_contract(self) -> None:
        with h5py.File(self.path, "r") as handle:
            for sample_id in (self.sample_ids[0], self.sample_ids[-1]):
                group_name = f"Sample_{sample_id}"
                if group_name not in handle:
                    raise KeyError(f"missing group {group_name} in {self.path}")
                group = handle[group_name]
                if self.input_key not in group or self.output_key not in group:
                    raise KeyError(
                        f"{group_name} must contain {self.input_key!r} and {self.output_key!r}"
                    )
                input_shape = tuple(group[self.input_key].shape)
                output_shape = tuple(group[self.output_key].shape)
                if len(input_shape) != 2 or input_shape != output_shape:
                    raise ValueError(
                        f"invalid field shapes in {group_name}: {input_shape}, {output_shape}"
                    )

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        sample_id = self.sample_ids[index]
        group = self._file()[f"Sample_{sample_id}"]
        input_array = np.asarray(group[self.input_key], dtype=np.float32)
        output_array = np.asarray(group[self.output_key], dtype=np.float32)
        if input_array.shape != output_array.shape or input_array.ndim != 2:
            raise ValueError(f"invalid shapes for Sample_{sample_id}")
        if not np.isfinite(input_array).all() or not np.isfinite(output_array).all():
            raise ValueError(f"nonfinite field values in Sample_{sample_id}")
        input_tensor = torch.from_numpy(input_array.copy()).unsqueeze(0)
        output_tensor = torch.from_numpy(output_array.copy()).unsqueeze(0)
        return (
            self.normalizer.normalize_input(input_tensor),
            self.normalizer.normalize_output(output_tensor),
            sample_id,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        # h5py modules may already be partially torn down during interpreter
        # shutdown.  Explicit ``close`` remains available for normal control
        # flow; finalization must never emit a spurious exception.
        try:
            self.close()
        except Exception:
            self._handle = None


def relative_l1_per_sample(prediction: Tensor, target: Tensor, epsilon: float) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} versus {target.shape}"
        )
    reduce_dims = tuple(range(1, prediction.ndim))
    numerator = torch.sum(torch.abs(prediction - target), dim=reduce_dims)
    denominator = torch.sum(torch.abs(target), dim=reduce_dims).clamp_min(epsilon)
    return numerator / denominator


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".pth", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz_save(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
