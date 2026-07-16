from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import scipy.io
import torch


REQUIRED_FIELDS = {
    "x_f1",
    "y_f1",
    "x_f2",
    "y_f2",
    "x_f3",
    "y_f3",
    "xi1",
    "yi1",
    "xi2",
    "yi2",
    "xb",
    "yb",
    "ub",
    "u_exact",
    "u_exact1",
    "u_exact2",
    "u_exact3",
}


def load_mat_data(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"XPINN MATLAB data not found: {path}")
    data = scipy.io.loadmat(path)
    missing = REQUIRED_FIELDS.difference(data)
    if missing:
        raise ValueError(f"MATLAB data is missing fields: {sorted(missing)}")
    return data


def column(data: Mapping, key: str) -> np.ndarray:
    return np.asarray(data[key], dtype=np.float64).reshape(-1, 1)


def sample_indices(
    generator: np.random.Generator, total_size: int, sample_size: int, name: str
) -> np.ndarray:
    if sample_size <= 0:
        raise ValueError(f"{name} sample size must be positive")
    if sample_size > total_size:
        raise ValueError(
            f"{name} sample size {sample_size} exceeds available points {total_size}"
        )
    return generator.choice(total_size, sample_size, replace=False)


def tensor(
    values: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype, device=device).clone().requires_grad_(
        requires_grad
    )


def paired_sample(
    data: Mapping,
    x_key: str,
    y_key: str,
    sample_size: int,
    generator: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = column(data, x_key)
    y = column(data, y_key)
    if x.shape != y.shape:
        raise ValueError(f"coordinate shape mismatch for {name}: {x.shape} and {y.shape}")
    indices = sample_indices(generator, x.shape[0], sample_size, name)
    return (
        tensor(x[indices], device, dtype, requires_grad=True),
        tensor(y[indices], device, dtype, requires_grad=True),
    )


def build_training_batch(
    data: Mapping,
    sample_counts: Mapping[str, int],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    generator = np.random.default_rng(seed)
    x1, y1 = paired_sample(
        data,
        "x_f1",
        "y_f1",
        int(sample_counts["residual_1"]),
        generator,
        device,
        dtype,
        "residual_1",
    )
    x2, y2 = paired_sample(
        data,
        "x_f2",
        "y_f2",
        int(sample_counts["residual_2"]),
        generator,
        device,
        dtype,
        "residual_2",
    )
    x3, y3 = paired_sample(
        data,
        "x_f3",
        "y_f3",
        int(sample_counts["residual_3"]),
        generator,
        device,
        dtype,
        "residual_3",
    )
    xi1, yi1 = paired_sample(
        data,
        "xi1",
        "yi1",
        int(sample_counts["interface_1"]),
        generator,
        device,
        dtype,
        "interface_1",
    )
    xi2, yi2 = paired_sample(
        data,
        "xi2",
        "yi2",
        int(sample_counts["interface_2"]),
        generator,
        device,
        dtype,
        "interface_2",
    )

    boundary_x = column(data, "xb")
    boundary_y = column(data, "yb")
    boundary_values = column(data, "ub")
    if boundary_x.shape != boundary_y.shape or boundary_x.shape != boundary_values.shape:
        raise ValueError("boundary coordinate and value shapes do not match")
    boundary_indices = sample_indices(
        generator,
        boundary_x.shape[0],
        int(sample_counts["boundary"]),
        "boundary",
    )
    return {
        "xb": tensor(boundary_x[boundary_indices], device, dtype),
        "yb": tensor(boundary_y[boundary_indices], device, dtype),
        "ub": tensor(boundary_values[boundary_indices], device, dtype),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x3": x3,
        "y3": y3,
        "xi1": xi1,
        "yi1": yi1,
        "xi2": xi2,
        "yi2": yi2,
    }


def build_evaluation_points(
    data: Mapping, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    points = {}
    for domain in (1, 2, 3):
        x = column(data, f"x_f{domain}")
        y = column(data, f"y_f{domain}")
        if x.shape != y.shape:
            raise ValueError(f"evaluation coordinate mismatch in domain {domain}")
        points[f"xy{domain}"] = tensor(np.hstack((x, y)), device, dtype)
    return points


def exact_subdomain_values(
    data: Mapping, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        tensor(column(data, f"u_exact{domain}"), device, dtype)
        for domain in (1, 2, 3)
    )


def combined_coordinates(data: Mapping) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([column(data, f"x_f{domain}").reshape(-1) for domain in (1, 2, 3)])
    y = np.concatenate([column(data, f"y_f{domain}").reshape(-1) for domain in (1, 2, 3)])
    return x, y


def combined_exact_solution(data: Mapping) -> np.ndarray:
    exact = column(data, "u_exact").reshape(-1)
    expected_size = sum(column(data, f"x_f{domain}").size for domain in (1, 2, 3))
    if exact.size != expected_size:
        raise ValueError(
            f"combined exact solution has {exact.size} values, expected {expected_size}"
        )
    return exact
