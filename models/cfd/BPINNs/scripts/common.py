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


def exact_solution(values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if torch.is_tensor(values):
        return torch.sin(torch.pi * values)
    return np.sin(np.pi * values)


def build_laplace_data(
    config: Mapping,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    domain = config["domain"]
    if len(domain) != 2 or float(domain[0]) >= float(domain[1]):
        raise ValueError("data.domain must contain increasing lower and upper bounds")
    lower, upper = float(domain[0]), float(domain[1])
    n_solution = int(config["n_sol"])
    n_pde = int(config["n_pde"])
    test_resolution = int(config["test_res"])
    if min(n_solution, n_pde, test_resolution) <= 0:
        raise ValueError("n_sol, n_pde, and test_res must be positive")
    noise_std = float(config["noise_std"])
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    generator = np.random.default_rng(seed)
    x_solution = generator.uniform(lower, upper, (n_solution, 1))
    u_solution = exact_solution(x_solution)
    if noise_std:
        u_solution = u_solution + noise_std * generator.standard_normal(u_solution.shape)
    x_pde = generator.uniform(lower, upper, (n_pde, 1))
    x_boundary = np.array([[lower], [upper]])
    u_boundary = exact_solution(x_boundary)
    x_test = np.linspace(lower, upper, test_resolution)[:, None]
    u_test = exact_solution(x_test)

    return {
        "x_solution": torch.as_tensor(x_solution, dtype=dtype, device=device),
        "u_solution": torch.as_tensor(u_solution, dtype=dtype, device=device),
        "x_pde": torch.as_tensor(x_pde, dtype=dtype, device=device),
        "x_boundary": torch.as_tensor(x_boundary, dtype=dtype, device=device),
        "u_boundary": torch.as_tensor(u_boundary, dtype=dtype, device=device),
        "x_test": torch.as_tensor(x_test, dtype=dtype, device=device),
        "u_test": torch.as_tensor(u_test, dtype=dtype, device=device),
    }


def relative_l2(prediction: np.ndarray | torch.Tensor, reference: np.ndarray | torch.Tensor) -> float:
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.detach().cpu().numpy()
    prediction_array = np.asarray(prediction).reshape(-1)
    reference_array = np.asarray(reference).reshape(-1)
    return float(
        np.linalg.norm(prediction_array - reference_array)
        / (np.linalg.norm(reference_array) + 1.0e-12)
    )


def checkpoint_state(checkpoint: Mapping) -> tuple[Mapping[str, torch.Tensor], dict]:
    if "model_state" in checkpoint:
        return checkpoint["model_state"], dict(checkpoint)
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, {}
    raise ValueError("checkpoint contains no valid BPINN model state")
