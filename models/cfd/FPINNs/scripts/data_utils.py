from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import scipy.io
import torch


def load_allen_cahn(path: Path) -> dict[str, np.ndarray | tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Allen-Cahn data not found: {path}")
    data = scipy.io.loadmat(path)
    missing = {"tt", "x", "uu"}.difference(data)
    if missing:
        raise ValueError(f"Allen-Cahn data is missing fields: {sorted(missing)}")
    time_axis = np.asarray(data["tt"], dtype=np.float64).reshape(-1)
    space_axis = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    raw_solution = np.real(np.asarray(data["uu"]))
    expected_shape = (space_axis.size, time_axis.size)
    if raw_solution.shape != expected_shape:
        raise ValueError(
            f"uu shape must be {expected_shape} in [x,t] order, got {raw_solution.shape}"
        )
    exact_grid = raw_solution.T
    mesh_x, mesh_t = np.meshgrid(space_axis, time_axis, indexing="xy")
    coordinates = np.column_stack((mesh_x.ravel(), mesh_t.ravel()))
    exact = exact_grid.reshape(-1, 1)
    return {
        "time": time_axis,
        "space": space_axis,
        "coordinates": coordinates,
        "exact": exact,
        "grid_shape": exact_grid.shape,
    }


def sample_training_data(
    dataset: Mapping,
    n_train: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = np.asarray(dataset["coordinates"])
    exact = np.asarray(dataset["exact"])
    if n_train <= 0 or n_train > coordinates.shape[0]:
        raise ValueError(
            f"n_train must be between 1 and {coordinates.shape[0]}, got {n_train}"
        )
    generator = np.random.default_rng(seed)
    indices = generator.choice(coordinates.shape[0], n_train, replace=False)
    return (
        torch.as_tensor(coordinates[indices], dtype=dtype, device=device),
        torch.as_tensor(exact[indices], dtype=dtype, device=device),
    )


def batched_predict(
    model: torch.nn.Module,
    coordinates: np.ndarray,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coordinates.shape[0], batch_size):
            batch = torch.as_tensor(
                coordinates[start : start + batch_size], dtype=dtype, device=device
            )
            predictions.append(model.predict_u(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0)
