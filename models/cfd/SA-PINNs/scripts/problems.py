from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from model.sa_pinn import Burgers2D, Equation, Helmholtz2D, Laplace1D


CASES = ("laplace", "helmholtz", "burgers")


def exact_solution(case: str, coordinates: np.ndarray) -> np.ndarray | None:
    if case == "laplace":
        return np.sin(np.pi * coordinates[:, 0:1])
    if case == "helmholtz":
        return np.sin(np.pi * coordinates[:, 0:1]) * np.sin(
            4.0 * np.pi * coordinates[:, 1:2]
        )
    if case == "burgers":
        return None
    raise ValueError(f"unsupported case: {case}")


def build_equation(case: str, data_config: Mapping) -> Equation:
    if case == "laplace":
        return Laplace1D()
    if case == "helmholtz":
        return Helmholtz2D(float(data_config["wave_number"]))
    if case == "burgers":
        return Burgers2D(float(data_config["viscosity"]))
    raise ValueError(f"unsupported case: {case}")


def generate_data(case: str, config: Mapping, seed: int) -> dict[str, np.ndarray | None]:
    generator = np.random.default_rng(seed)
    n_pde = int(config["n_pde"])
    n_solution = int(config["n_sol"])
    n_boundary = int(config["n_bnd"])
    test_resolution = int(config["test_res"])
    if min(n_pde, n_solution, n_boundary, test_resolution) <= 0:
        raise ValueError("all data counts must be positive")

    if case == "laplace":
        lower, upper = map(float, config["domain"])
        x_pde = generator.uniform(lower, upper, (n_pde, 1))
        x_boundary = np.array([[lower], [upper]], dtype=np.float64)
        u_boundary = exact_solution(case, x_boundary)
        x_data = np.linspace(lower, upper, n_solution)[:, None]
        u_data = exact_solution(case, x_data)
        noise_std = float(config.get("noise_std", 0.0))
        if noise_std:
            u_data = u_data + noise_std * generator.standard_normal(u_data.shape)
        x_test = np.linspace(lower, upper, test_resolution)[:, None]
        test_shape = (test_resolution,)
    elif case == "helmholtz":
        x_lower, x_upper = map(float, config["domain_x"])
        y_lower, y_upper = map(float, config["domain_y"])
        x_pde = np.column_stack(
            (
                generator.uniform(x_lower, x_upper, n_pde),
                generator.uniform(y_lower, y_upper, n_pde),
            )
        )
        if n_boundary < 4 or n_boundary % 4:
            raise ValueError("Helmholtz n_bnd must be divisible by 4 and at least 4")
        per_edge = n_boundary // 4
        x_axis = np.linspace(x_lower, x_upper, per_edge)
        y_axis = np.linspace(y_lower, y_upper, per_edge)
        x_boundary = np.vstack(
            (
                np.column_stack((x_axis, np.full(per_edge, y_lower))),
                np.column_stack((x_axis, np.full(per_edge, y_upper))),
                np.column_stack((np.full(per_edge, x_lower), y_axis)),
                np.column_stack((np.full(per_edge, x_upper), y_axis)),
            )
        )
        u_boundary = exact_solution(case, x_boundary)
        x_data = np.column_stack(
            (
                generator.uniform(x_lower, x_upper, n_solution),
                generator.uniform(y_lower, y_upper, n_solution),
            )
        )
        u_data = exact_solution(case, x_data)
        grid_x = np.linspace(x_lower, x_upper, test_resolution)
        grid_y = np.linspace(y_lower, y_upper, test_resolution)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="xy")
        x_test = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
        test_shape = mesh_x.shape
    elif case == "burgers":
        x_lower, x_upper = map(float, config["domain_x"])
        t_lower, t_upper = map(float, config["domain_t"])
        x_pde = np.column_stack(
            (
                generator.uniform(x_lower, x_upper, n_pde),
                generator.uniform(t_lower, t_upper, n_pde),
            )
        )
        if n_boundary < 2:
            raise ValueError("Burgers n_bnd must be at least 2")
        left_count = n_boundary // 2
        right_count = n_boundary - left_count
        left_time = generator.uniform(t_lower, t_upper, left_count)
        right_time = generator.uniform(t_lower, t_upper, right_count)
        x_boundary = np.vstack(
            (
                np.column_stack((np.full(left_count, x_lower), left_time)),
                np.column_stack((np.full(right_count, x_upper), right_time)),
            )
        )
        u_boundary = np.zeros((n_boundary, 1))
        initial_x = generator.uniform(x_lower, x_upper, n_solution)
        x_data = np.column_stack((initial_x, np.full(n_solution, t_lower)))
        u_data = -np.sin(np.pi * initial_x)[:, None]
        grid_x = np.linspace(x_lower, x_upper, test_resolution)
        grid_t = np.linspace(t_lower, t_upper, test_resolution)
        mesh_x, mesh_t = np.meshgrid(grid_x, grid_t, indexing="xy")
        x_test = np.column_stack((mesh_x.ravel(), mesh_t.ravel()))
        test_shape = mesh_x.shape
    else:
        raise ValueError(f"unsupported case: {case}")

    return {
        "x_pde": np.asarray(x_pde, dtype=np.float64),
        "x_boundary": np.asarray(x_boundary, dtype=np.float64),
        "u_boundary": np.asarray(u_boundary, dtype=np.float64),
        "x_data": np.asarray(x_data, dtype=np.float64),
        "u_data": np.asarray(u_data, dtype=np.float64),
        "x_test": np.asarray(x_test, dtype=np.float64),
        "u_exact": exact_solution(case, np.asarray(x_test, dtype=np.float64)),
        "test_shape": test_shape,
    }


def to_tensors(
    data: Mapping[str, np.ndarray | None],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | None]:
    return {
        key: (
            torch.as_tensor(value, dtype=dtype, device=device)
            if isinstance(value, np.ndarray)
            else None
        )
        for key, value in data.items()
        if key != "test_shape"
    }


def point_counts(data: Mapping[str, np.ndarray | None]) -> dict[str, int]:
    return {
        "pde": int(data["x_pde"].shape[0]),
        "boundary": int(data["x_boundary"].shape[0]),
        "data": int(data["x_data"].shape[0]) if data.get("x_data") is not None else 0,
    }


def relative_l2(prediction: np.ndarray, reference: np.ndarray | None) -> float | None:
    if reference is None:
        return None
    return float(
        np.linalg.norm(prediction.reshape(-1) - reference.reshape(-1))
        / (np.linalg.norm(reference.reshape(-1)) + 1.0e-12)
    )
