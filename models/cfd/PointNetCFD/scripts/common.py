"""Shared data, metric, configuration, and serialization utilities."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


PAPER_SAMPLE_COUNT = 2595
AVAILABLE_SAMPLE_COUNT = 2215


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and minimally validate the experiment YAML."""
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    required_sections = ("experiment", "paths", "data", "model", "training", "evaluation")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Configuration is missing sections: {missing}")

    data_config = config["data"]
    if data_config.get("coordinate_normalization") != "none":
        raise ValueError("Paper fidelity requires coordinate_normalization: none")
    if data_config.get("target_normalization") != "train_minmax":
        raise ValueError("Paper fidelity requires target_normalization: train_minmax")
    if list(data_config.get("target_names", [])) != ["u", "v", "p"]:
        raise ValueError("Model target order must be [u, v, p]")
    if config["training"].get("scheduler") != "none":
        raise ValueError("The paper does not specify a learning-rate scheduler")
    return config


def resolve_path(project_root: Path, configured_path: str) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def configured_paths(config: Mapping[str, Any], project_root: Path) -> Dict[str, Path]:
    """Resolve all data and output paths without consulting environment variables."""
    paths = config["paths"]
    data_dir = resolve_path(project_root, str(paths["data_dir"]))
    return {
        "data": data_dir / str(paths["data_file"]),
        "train_indices": data_dir / str(paths["train_indices"]),
        "validation_indices": data_dir / str(paths["validation_indices"]),
        "test_indices": data_dir / str(paths["test_indices"]),
        "checkpoint": resolve_path(project_root, str(paths["checkpoint"])),
        "results_dir": resolve_path(project_root, str(paths["results_dir"])),
    }


def set_deterministic_seed(seed: int) -> None:
    """Seed every RNG used by this project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def choose_device(requested: str) -> torch.device:
    """Resolve ``auto``, CPU, or an explicit CUDA device."""
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def _load_index_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Split index file not found: {path}")
    indices = np.load(path, allow_pickle=False)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"Split indices must be a one-dimensional integer array: {path}")
    return np.asarray(indices, dtype=np.int64)


def load_data_and_splits(
    config: Mapping[str, Any], project_root: Path
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Path]]:
    """Load the CFD array read-only and validate the fixed supplied splits."""
    paths = configured_paths(config, project_root)
    if not paths["data"].is_file():
        raise FileNotFoundError(f"CFD data file not found: {paths['data']}")
    data = np.load(paths["data"], mmap_mode="r", allow_pickle=False)
    expected_points = int(config["data"]["num_points"])
    expected_channels = len(config["data"]["source_channels"])
    if data.ndim != 3 or data.shape[1:] != (expected_points, expected_channels):
        raise ValueError(
            f"Expected CFD array [cases,{expected_points},{expected_channels}], "
            f"got {data.shape}"
        )
    if data.dtype != np.float32:
        raise ValueError(f"Expected float32 CFD data, got {data.dtype}")
    if not np.isfinite(data).all():
        raise ValueError("CFD data contains NaN or infinite values")

    splits = {
        "train": _load_index_array(paths["train_indices"]),
        "validation": _load_index_array(paths["validation_indices"]),
        "test": _load_index_array(paths["test_indices"]),
    }
    total_cases = int(data.shape[0])
    for name, indices in splits.items():
        if indices.size == 0:
            raise ValueError(f"Split {name} is empty")
        if indices.min() < 0 or indices.max() >= total_cases:
            raise ValueError(f"Split {name} contains out-of-range indices")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Split {name} contains duplicate indices")
    split_sets = {name: set(values.tolist()) for name, values in splits.items()}
    if split_sets["train"] & split_sets["validation"]:
        raise ValueError("Training and validation splits overlap")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Training and test splits overlap")
    if split_sets["validation"] & split_sets["test"]:
        raise ValueError("Validation and test splits overlap")
    covered = split_sets["train"] | split_sets["validation"] | split_sets["test"]
    if covered != set(range(total_cases)):
        raise ValueError("Fixed splits do not cover the dataset exactly")
    return data, splits, paths


def fit_target_minmax(
    data: np.ndarray, train_indices: np.ndarray, target_indices: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-variable extrema using the training split and no other cases."""
    training_cases = np.asarray(data[train_indices], dtype=np.float32)
    training_targets = np.take(training_cases, target_indices, axis=-1)
    target_min = training_targets.min(axis=(0, 1)).astype(np.float32)
    target_max = training_targets.max(axis=(0, 1)).astype(np.float32)
    if not np.isfinite(target_min).all() or not np.isfinite(target_max).all():
        raise ValueError("Target normalization extrema are not finite")
    if np.any(target_max <= target_min):
        raise ValueError(
            f"Every target must have positive min-max span: min={target_min}, max={target_max}"
        )
    return target_min, target_max


class PointCFDDataset(Dataset):
    """A fixed case-index view of the supplied PointCFD NumPy array."""

    def __init__(
        self,
        data: np.ndarray,
        case_indices: np.ndarray,
        input_indices: Sequence[int],
        target_indices: Sequence[int],
        target_min: np.ndarray,
        target_max: np.ndarray,
    ) -> None:
        self.data = data
        self.case_indices = np.asarray(case_indices, dtype=np.int64)
        self.input_indices = tuple(int(index) for index in input_indices)
        self.target_indices = tuple(int(index) for index in target_indices)
        self.target_min = np.asarray(target_min, dtype=np.float32)
        self.target_max = np.asarray(target_max, dtype=np.float32)
        if self.target_min.shape != (len(self.target_indices),):
            raise ValueError("target_min has the wrong shape")
        if self.target_max.shape != self.target_min.shape:
            raise ValueError("target_max has the wrong shape")
        self.target_span = self.target_max - self.target_min
        if np.any(self.target_span <= 0):
            raise ValueError("Target min-max span must be positive")

    def __len__(self) -> int:
        return int(self.case_indices.size)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor]:
        case_index = int(self.case_indices[item])
        sample = self.data[case_index]
        coordinates = np.ascontiguousarray(sample[:, self.input_indices], dtype=np.float32)
        targets = np.ascontiguousarray(sample[:, self.target_indices], dtype=np.float32)
        normalized = np.ascontiguousarray(
            (targets - self.target_min) / self.target_span, dtype=np.float32
        )
        return (
            torch.from_numpy(coordinates),
            torch.from_numpy(normalized),
            torch.tensor(case_index, dtype=torch.int64),
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker if num_workers else None,
    )


def inverse_target_minmax(
    normalized: np.ndarray, target_min: np.ndarray, target_max: np.ndarray
) -> np.ndarray:
    return normalized * (target_max - target_min) + target_min


def compute_field_metrics(
    predictions_normalized: np.ndarray,
    targets_normalized: np.ndarray,
    target_min: np.ndarray,
    target_max: np.ndarray,
    target_names: Sequence[str],
    relative_l2_epsilon: float,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Compute paper-style field error metrics after inverse normalization."""
    predictions_normalized = np.asarray(predictions_normalized, dtype=np.float32)
    targets_normalized = np.asarray(targets_normalized, dtype=np.float32)
    if predictions_normalized.shape != targets_normalized.shape:
        raise ValueError("Prediction and target shapes do not match")
    if predictions_normalized.ndim != 3:
        raise ValueError("Evaluation arrays must be [cases, points, variables]")
    if predictions_normalized.shape[-1] != len(target_names):
        raise ValueError("Target names do not match the evaluated variables")
    if relative_l2_epsilon <= 0:
        raise ValueError("relative_l2_epsilon must be positive")

    normalized_error = predictions_normalized - targets_normalized
    normalized_mse = float(np.mean(np.square(normalized_error), dtype=np.float64))
    predictions = inverse_target_minmax(predictions_normalized, target_min, target_max)
    targets = inverse_target_minmax(targets_normalized, target_min, target_max)
    error = predictions - targets

    rmse: Dict[str, float] = {}
    relative_l2: Dict[str, Dict[str, float]] = {}
    for channel, name in enumerate(target_names):
        channel_error = error[:, :, channel].astype(np.float64)
        channel_target = targets[:, :, channel].astype(np.float64)
        rmse[str(name)] = float(np.sqrt(np.mean(np.square(channel_error))))
        numerator = np.linalg.norm(channel_error, axis=1)
        denominator = np.maximum(
            np.linalg.norm(channel_target, axis=1), relative_l2_epsilon
        )
        per_case = numerator / denominator
        relative_l2[str(name)] = {
            "mean": float(np.mean(per_case)),
            "max": float(np.max(per_case)),
            "min": float(np.min(per_case)),
        }
    metrics = {
        "normalized_mse": normalized_mse,
        "rmse": rmse,
        "relative_l2": relative_l2,
    }
    return metrics, predictions.astype(np.float32), targets.astype(np.float32)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_min: np.ndarray,
    target_max: np.ndarray,
    target_names: Sequence[str],
    relative_l2_epsilon: float,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Evaluate a complete split and retain arrays needed for reporting."""
    model.eval()
    coordinates_batches = []
    prediction_batches = []
    target_batches = []
    index_batches = []
    with torch.no_grad():
        for coordinates, targets, case_indices in loader:
            predictions = model(coordinates.to(device, non_blocking=True))
            coordinates_batches.append(coordinates.numpy())
            prediction_batches.append(predictions.cpu().numpy())
            target_batches.append(targets.numpy())
            index_batches.append(case_indices.numpy())
    if not prediction_batches:
        raise ValueError("Evaluation loader produced no batches")
    coordinates_array = np.concatenate(coordinates_batches, axis=0)
    predictions_normalized = np.concatenate(prediction_batches, axis=0)
    targets_normalized = np.concatenate(target_batches, axis=0)
    case_indices_array = np.concatenate(index_batches, axis=0)
    metrics, predictions, targets = compute_field_metrics(
        predictions_normalized,
        targets_normalized,
        target_min,
        target_max,
        target_names,
        relative_l2_epsilon,
    )
    arrays = {
        "coordinates": coordinates_array.astype(np.float32),
        "predictions": predictions,
        "targets": targets,
        "predictions_normalized": predictions_normalized.astype(np.float32),
        "targets_normalized": targets_normalized.astype(np.float32),
        "case_indices": case_indices_array.astype(np.int64),
    }
    return metrics, arrays


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("JSON output cannot contain NaN or Infinity")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write strict, human-readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write a compressed NumPy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    return checkpoint


def rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_metadata_matches(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    target_min: np.ndarray,
    target_max: np.ndarray,
) -> None:
    """Reject silent channel or normalization changes on resume."""
    expected = {
        "source_channels": list(config["data"]["source_channels"]),
        "input_names": list(config["data"]["input_names"]),
        "target_names": list(config["data"]["target_names"]),
    }
    for key, value in expected.items():
        if list(checkpoint.get(key, [])) != value:
            raise ValueError(f"Checkpoint {key} metadata does not match the configuration")
    if not np.allclose(np.asarray(checkpoint.get("target_min")), target_min, rtol=0, atol=0):
        raise ValueError("Checkpoint target_min differs from the current training split")
    if not np.allclose(np.asarray(checkpoint.get("target_max")), target_max, rtol=0, atol=0):
        raise ValueError("Checkpoint target_max differs from the current training split")


def selected_indices(indices: np.ndarray, maximum_cases: Optional[int]) -> np.ndarray:
    if maximum_cases is None:
        return indices
    if maximum_cases <= 0:
        raise ValueError("maximum_cases must be positive")
    return indices[: min(maximum_cases, indices.size)]


def prepare_datasets(
    config: Mapping[str, Any],
    project_root: Path,
    smoke_test: bool = False,
) -> Tuple[Dict[str, PointCFDDataset], np.ndarray, np.ndarray, Dict[str, Path], Dict[str, int]]:
    """Build all split datasets under one normalization contract."""
    data, splits, paths = load_data_and_splits(config, project_root)
    if smoke_test:
        active_splits = {
            "train": selected_indices(splits["train"], 4),
            "validation": selected_indices(splits["validation"], 2),
            "test": selected_indices(splits["test"], 2),
        }
    else:
        active_splits = splits
    target_indices = tuple(int(i) for i in config["data"]["target_indices"])
    target_min, target_max = fit_target_minmax(data, active_splits["train"], target_indices)
    datasets = {
        name: PointCFDDataset(
            data,
            indices,
            config["data"]["input_indices"],
            target_indices,
            target_min,
            target_max,
        )
        for name, indices in active_splits.items()
    }
    counts = {name: len(dataset) for name, dataset in datasets.items()}
    return datasets, target_min, target_max, paths, counts
