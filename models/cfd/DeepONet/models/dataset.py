"""Synthetic datasets for the four main experiments in arXiv:1910.03193.

All equations and split rules follow the paper.  The numerical choices that the
paper omits are configurable and documented in ``config/config.yaml``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.integrate import cumulative_trapezoid, solve_ivp
    from scipy.interpolate import CubicSpline
except ImportError:  # Deferred error keeps model-only imports usable.
    cumulative_trapezoid = None
    solve_ivp = None
    CubicSpline = None


def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update ``base`` without mutating the caller's mapping."""

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = deep_update(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def resolve_config(config: Mapping[str, Any], smoke_test: bool = False) -> Dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    smoke_override = resolved.pop("smoke_test", {})
    if smoke_test:
        resolved = deep_update(resolved, smoke_override)
    resolved.setdefault("project", {})["paper_scale"] = not smoke_test
    return resolved


class OperatorDataset(Dataset):
    """Triplets with compact storage for repeated PDE branch functions."""

    def __init__(
        self,
        branch_functions: np.ndarray,
        trunk: np.ndarray,
        target: np.ndarray,
        function_index: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        branch_functions = np.asarray(branch_functions, dtype=np.float32)
        trunk = np.asarray(trunk, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if function_index is None:
            function_index = np.arange(len(trunk), dtype=np.int64)
        function_index = np.asarray(function_index, dtype=np.int64)
        if branch_functions.ndim != 2 or trunk.ndim != 2 or target.ndim != 2:
            raise ValueError("branch, trunk and target arrays must all have rank two")
        if target.shape[1] != 1 or len(trunk) != len(target) or len(trunk) != len(function_index):
            raise ValueError("trunk, target and function_index lengths must agree")
        if len(function_index) and (
            function_index.min() < 0 or function_index.max() >= len(branch_functions)
        ):
            raise ValueError("function_index refers outside branch_functions")
        self.branch_functions = branch_functions
        self.trunk = trunk
        self.target = target
        self.function_index = function_index
        self.metadata = dict(metadata or {})

    def __len__(self) -> int:
        return len(self.trunk)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        function_id = self.function_index[index]
        return (
            torch.from_numpy(self.branch_functions[function_id]),
            torch.from_numpy(self.trunk[index]),
            torch.from_numpy(self.target[index]),
        )

    def expanded_branch(self) -> np.ndarray:
        return self.branch_functions[self.function_index]


class FunctionSpaceSampler:
    """GRF or Chebyshev function sampler on a reusable fine grid."""

    def __init__(self, config: Mapping[str, Any], domain_end: float) -> None:
        self.config = dict(config)
        self.domain_end = float(domain_end)
        self.grid_size = int(self.config["grf_grid_size"])
        self.grid = np.linspace(0.0, self.domain_end, self.grid_size, dtype=np.float64)
        self.kind = str(self.config.get("type", "grf")).lower()
        self._cholesky: np.ndarray | None = None
        if self.kind == "grf":
            length_scale = float(self.config["length_scale"])
            distances = self.grid[:, None] - self.grid[None, :]
            covariance = np.exp(-(distances**2) / (2.0 * length_scale**2))
            jitter = float(self.config.get("jitter", 1.0e-13))
            identity = np.eye(self.grid_size, dtype=np.float64)
            for attempt in range(6):
                try:
                    self._cholesky = np.linalg.cholesky(covariance + jitter * identity)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            if self._cholesky is None:
                raise np.linalg.LinAlgError("GRF covariance Cholesky failed after jitter fallback")
        elif self.kind != "chebyshev":
            raise ValueError(f"Unsupported function space {self.kind!r}")

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "grf":
            standard_normal = rng.standard_normal((self.grid_size, count))
            return (self._cholesky @ standard_normal).T
        cheb = self.config.get("chebyshev", {})
        degree = int(cheb.get("degree", 10))
        bound = float(cheb.get("coefficient_bound", 1.0))
        coefficients = rng.uniform(-bound, bound, size=(count, degree + 1))
        mapped_grid = 2.0 * self.grid / self.domain_end - 1.0
        return np.stack(
            [np.polynomial.chebyshev.chebval(mapped_grid, row) for row in coefficients], axis=0
        )

    def interpolate(self, values: np.ndarray, points: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        method = str(self.config.get("interpolation", "cubic")).lower()
        if method == "cubic":
            _require_scipy("cubic GRF interpolation")
            return np.asarray(CubicSpline(self.grid, values, axis=-1)(points))
        if method != "linear":
            raise ValueError(f"Unsupported interpolation method {method!r}")
        if values.ndim == 1:
            return np.interp(points, self.grid, values)
        return np.stack([np.interp(points, self.grid, row) for row in values], axis=0)


def _require_scipy(operation: str) -> None:
    if solve_ivp is None or CubicSpline is None or cumulative_trapezoid is None:
        raise ImportError(f"SciPy is required for {operation}; install it in the execution environment")


def _rowwise_linear_interpolation(
    grid: np.ndarray, values: np.ndarray, points: np.ndarray
) -> np.ndarray:
    points = np.clip(np.asarray(points), grid[0], grid[-1])
    right = np.searchsorted(grid, points, side="right")
    right = np.clip(right, 1, len(grid) - 1)
    left = right - 1
    fraction = (points - grid[left]) / (grid[right] - grid[left])
    rows = np.arange(len(points))
    return values[rows, left] * (1.0 - fraction) + values[rows, right] * fraction


def solve_antiderivative(
    input_grid: np.ndarray, input_values: np.ndarray, query_points: np.ndarray
) -> np.ndarray:
    _require_scipy("antiderivative reference generation")
    integral = cumulative_trapezoid(input_values, input_grid, axis=-1, initial=0.0)
    return _rowwise_linear_interpolation(input_grid, integral, query_points)


def solve_nonlinear_ode(
    input_grid: np.ndarray,
    input_values: np.ndarray,
    query_points: np.ndarray,
    solver_config: Mapping[str, Any],
) -> np.ndarray:
    _require_scipy("nonlinear ODE reference generation")
    query_points = np.asarray(query_points, dtype=np.float64)
    if query_points.ndim != 1:
        raise ValueError("query_points must be one-dimensional")
    if not len(query_points):
        return np.empty(0, dtype=np.float64)
    interpolant = CubicSpline(input_grid, input_values)
    maximum = float(np.max(query_points))
    if maximum == 0.0:
        return np.zeros_like(query_points)
    solution = solve_ivp(
        lambda x, state: -state**2 + interpolant(x),
        (0.0, maximum),
        np.zeros(1, dtype=np.float64),
        method=str(solver_config.get("method", "RK45")),
        rtol=float(solver_config.get("rtol", 1.0e-7)),
        atol=float(solver_config.get("atol", 1.0e-9)),
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"nonlinear ODE solve failed: {solution.message}")
    return np.asarray(solution.sol(query_points)[0])


def solve_pendulum(
    input_grid: np.ndarray,
    input_values: np.ndarray,
    query_points: np.ndarray,
    k: float,
    solver_config: Mapping[str, Any],
) -> np.ndarray:
    _require_scipy("pendulum reference generation")
    query_points = np.asarray(query_points, dtype=np.float64)
    if not len(query_points):
        return np.empty(0, dtype=np.float64)
    interpolant = CubicSpline(input_grid, input_values)
    maximum = float(np.max(query_points))
    if maximum == 0.0:
        return np.zeros_like(query_points)

    def right_hand_side(time: float, state: np.ndarray) -> np.ndarray:
        return np.asarray((state[1], -k * np.sin(state[0]) + interpolant(time)))

    solution = solve_ivp(
        right_hand_side,
        (0.0, maximum),
        np.zeros(2, dtype=np.float64),
        method=str(solver_config.get("method", "RK45")),
        rtol=float(solver_config.get("rtol", 1.0e-7)),
        atol=float(solver_config.get("atol", 1.0e-9)),
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"pendulum solve failed: {solution.message}")
    return np.asarray(solution.sol(query_points)[0])


def _solve_tridiagonal(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a nonsingular tridiagonal system."""

    lower = np.asarray(lower, dtype=np.float64).copy()
    diagonal = np.asarray(diagonal, dtype=np.float64).copy()
    upper = np.asarray(upper, dtype=np.float64).copy()
    rhs = np.asarray(rhs, dtype=np.float64).copy()
    for index in range(1, len(diagonal)):
        if abs(diagonal[index - 1]) < np.finfo(np.float64).eps:
            raise np.linalg.LinAlgError("zero pivot in tridiagonal solve")
        multiplier = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= multiplier * upper[index - 1]
        rhs[index] -= multiplier * rhs[index - 1]
    output = np.empty_like(rhs)
    output[-1] = rhs[-1] / diagonal[-1]
    for index in range(len(diagonal) - 2, -1, -1):
        output[index] = (rhs[index] - upper[index] * output[index + 1]) / diagonal[index]
    return output


def solve_diffusion_reaction(
    spatial_input: np.ndarray,
    *,
    diffusion: float,
    reaction: float,
    space_points: int,
    time_points: int,
    solver_config: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully implicit time stepping with second-order centered spatial differences."""

    x_grid = np.linspace(0.0, 1.0, int(space_points), dtype=np.float64)
    t_grid = np.linspace(0.0, 1.0, int(time_points), dtype=np.float64)
    source = np.asarray(spatial_input, dtype=np.float64)
    if source.shape != x_grid.shape:
        raise ValueError(f"spatial_input must have shape {(len(x_grid),)}, got {source.shape}")
    field = np.zeros((len(t_grid), len(x_grid)), dtype=np.float64)
    if len(x_grid) < 3 or len(t_grid) < 2:
        raise ValueError("PDE grid requires at least 3 spatial and 2 temporal points")
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]
    ratio = float(diffusion) * dt / (dx * dx)
    tolerance = float(solver_config.get("pde_newton_tolerance", 1.0e-10))
    max_iterations = int(solver_config.get("pde_newton_max_iterations", 20))
    interior_source = source[1:-1]
    interior_size = len(interior_source)
    off_diagonal = np.full(interior_size - 1, -ratio, dtype=np.float64)

    for time_index in range(1, len(t_grid)):
        old = field[time_index - 1, 1:-1]
        estimate = old.copy()
        for _ in range(max_iterations):
            padded = np.pad(estimate, (1, 1), mode="constant")
            laplacian_term = padded[:-2] - 2.0 * estimate + padded[2:]
            residual = (
                estimate
                - old
                - ratio * laplacian_term
                - dt * float(reaction) * estimate**2
                - dt * interior_source
            )
            diagonal = 1.0 + 2.0 * ratio - 2.0 * dt * float(reaction) * estimate
            update = _solve_tridiagonal(
                off_diagonal, diagonal, off_diagonal, -residual
            )
            estimate += update
            if np.max(np.abs(update)) <= tolerance:
                break
        else:
            raise RuntimeError(
                f"PDE Newton solve did not converge at time index {time_index}"
            )
        field[time_index, 1:-1] = estimate
    if not np.isfinite(field).all():
        raise FloatingPointError("PDE solver produced NaN or infinity")
    return x_grid, t_grid, field


def _generate_ode_like(
    config: Mapping[str, Any],
    experiment: str,
    count: int,
    seed: int,
) -> OperatorDataset:
    experiment_config = config["experiments"][experiment]
    function_config = config["function_space"]
    solver_config = config["solver_defaults"]
    rng = np.random.default_rng(seed)
    sampler = FunctionSpaceSampler(function_config, float(experiment_config["domain_end"]))
    sensors = np.linspace(
        0.0,
        float(experiment_config["domain_end"]),
        int(experiment_config["sensor_points"]),
        dtype=np.float64,
    )
    branch = np.empty((count, len(sensors)), dtype=np.float32)
    trunk = rng.uniform(0.0, float(experiment_config["domain_end"]), size=(count, 1))
    target = np.empty((count, 1), dtype=np.float32)
    chunk_size = int(function_config.get("generation_chunk_size", 128))
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        fine_values = sampler.sample(stop - start, rng)
        branch[start:stop] = sampler.interpolate(fine_values, sensors).astype(np.float32)
        local_queries = trunk[start:stop, 0]
        if experiment == "antiderivative":
            target[start:stop, 0] = solve_antiderivative(
                sampler.grid, fine_values, local_queries
            ).astype(np.float32)
            continue
        for local_index, values in enumerate(fine_values):
            query = np.asarray([local_queries[local_index]])
            if experiment == "nonlinear_ode":
                answer = solve_nonlinear_ode(sampler.grid, values, query, solver_config)
            elif experiment == "pendulum":
                answer = solve_pendulum(
                    sampler.grid,
                    values,
                    query,
                    float(experiment_config["k"]),
                    solver_config,
                )
            else:
                raise ValueError(f"Unsupported ODE-like experiment {experiment!r}")
            target[start + local_index, 0] = answer[0]
    return OperatorDataset(
        branch,
        trunk.astype(np.float32),
        target,
        metadata={"experiment": experiment, "seed": seed, "function_count": count},
    )


def _generate_pde(
    config: Mapping[str, Any],
    split: str,
    seed: int,
) -> OperatorDataset:
    experiment_config = config["experiments"]["diffusion_reaction"]
    if split == "train":
        function_count = int(experiment_config["train_functions"])
        points_per_function = int(experiment_config["points_per_function"])
    else:
        function_count = int(experiment_config["test_functions"])
        points_per_function = int(experiment_config["test_points_per_function"])
    rng = np.random.default_rng(seed)
    sampler = FunctionSpaceSampler(config["function_space"], 1.0)
    sensors = np.linspace(0.0, 1.0, int(experiment_config["sensor_points"]))
    branch = np.empty((function_count, len(sensors)), dtype=np.float32)
    total_points = function_count * points_per_function
    trunk = np.empty((total_points, 2), dtype=np.float32)
    target = np.empty((total_points, 1), dtype=np.float32)
    function_index = np.repeat(np.arange(function_count, dtype=np.int64), points_per_function)
    chunk_size = int(config["function_space"].get("generation_chunk_size", 128))
    cursor = 0
    for start in range(0, function_count, chunk_size):
        stop = min(function_count, start + chunk_size)
        fine_batch = sampler.sample(stop - start, rng)
        branch[start:stop] = sampler.interpolate(fine_batch, sensors).astype(np.float32)
        for values in fine_batch:
            spatial_grid = np.linspace(0.0, 1.0, int(experiment_config["space_points"]))
            spatial_input = sampler.interpolate(values, spatial_grid)
            x_grid, t_grid, field = solve_diffusion_reaction(
                spatial_input,
                diffusion=float(experiment_config["diffusion"]),
                reaction=float(experiment_config["reaction"]),
                space_points=int(experiment_config["space_points"]),
                time_points=int(experiment_config["time_points"]),
                solver_config=config["solver_defaults"],
            )
            grid_size = len(x_grid) * len(t_grid)
            flat_indices = rng.choice(
                grid_size,
                size=points_per_function,
                replace=points_per_function > grid_size,
            )
            time_indices, space_indices = np.divmod(flat_indices, len(x_grid))
            next_cursor = cursor + points_per_function
            trunk[cursor:next_cursor, 0] = x_grid[space_indices]
            trunk[cursor:next_cursor, 1] = t_grid[time_indices]
            target[cursor:next_cursor, 0] = field[time_indices, space_indices]
            cursor = next_cursor
    return OperatorDataset(
        branch,
        trunk,
        target,
        function_index,
        metadata={
            "experiment": "diffusion_reaction",
            "split": split,
            "seed": seed,
            "function_count": function_count,
            "points_per_function": points_per_function,
            "group_isolated": True,
        },
    )


def _fingerprint(config: Mapping[str, Any], experiment: str, split: str, seed: int) -> str:
    payload = json.dumps(
        {"config": config, "experiment": experiment, "split": split, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_path(
    config: Mapping[str, Any], project_root: Path, experiment: str, split: str, seed: int
) -> Tuple[Path, str]:
    fingerprint = _fingerprint(config, experiment, split, seed)
    relative_root = Path(config["paths"]["cache"])
    root = relative_root if relative_root.is_absolute() else project_root / relative_root
    return root / f"{experiment}_{split}_{fingerprint[:16]}.npz", fingerprint


def _save_cache(path: Path, dataset: OperatorDataset, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(dataset.metadata)
    metadata["fingerprint"] = fingerprint
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_path,
            branch_functions=dataset.branch_functions,
            trunk=dataset.trunk,
            target=dataset.target,
            function_index=dataset.function_index,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_cache(path: Path, fingerprint: str) -> OperatorDataset:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError(f"cache fingerprint mismatch for {path}")
        return OperatorDataset(
            payload["branch_functions"],
            payload["trunk"],
            payload["target"],
            payload["function_index"],
            metadata,
        )


def build_split(
    config: Mapping[str, Any],
    experiment: str,
    split: str,
    project_root: str | Path,
    *,
    use_cache: bool = True,
) -> OperatorDataset:
    """Build or load one independent train/test split."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if experiment not in config.get("experiments", {}):
        raise KeyError(f"Unknown experiment {experiment!r}")
    root = Path(project_root).resolve()
    base_seed = int(config["runtime"]["seed"])
    seed = base_seed + (0 if split == "train" else 100_000)
    path, fingerprint = _cache_path(config, root, experiment, split, seed)
    if use_cache and path.exists():
        return _load_cache(path, fingerprint)
    if experiment == "diffusion_reaction":
        dataset = _generate_pde(config, split, seed)
    else:
        size_key = "train_size" if split == "train" else "test_size"
        dataset = _generate_ode_like(
            config, experiment, int(config["experiments"][experiment][size_key]), seed
        )
    dataset.metadata.update(
        {
            "split": split,
            "paper_scale": bool(config["project"]["paper_scale"]),
            "fingerprint": fingerprint,
        }
    )
    if use_cache:
        _save_cache(path, dataset, fingerprint)
    return dataset


def build_datasets(
    config: Mapping[str, Any], experiment: str, project_root: str | Path
) -> Tuple[OperatorDataset, OperatorDataset]:
    train = build_split(config, experiment, "train", project_root)
    test = build_split(config, experiment, "test", project_root)
    return train, test


def analytic_input(name: str, coordinates: np.ndarray) -> np.ndarray:
    if name == "linear":
        return coordinates
    if name == "sin_pi":
        return np.sin(np.pi * coordinates)
    if name == "sin_2pi":
        return np.sin(2.0 * np.pi * coordinates)
    if name == "x_sin_2pi":
        return coordinates * np.sin(2.0 * np.pi * coordinates)
    raise KeyError(f"Unknown analytic input {name!r}")


def generate_ood_data(
    config: Mapping[str, Any], experiment: str, query_points: int | None = None
) -> Dict[str, np.ndarray]:
    if experiment == "diffusion_reaction":
        raise ValueError("Use generate_pde_grid_case for the PDE")
    experiment_config = config["experiments"][experiment]
    count = int(query_points or config["inference"]["ood_query_points"])
    domain_end = float(experiment_config["domain_end"])
    sensors = np.linspace(0.0, domain_end, int(experiment_config["sensor_points"]))
    queries = np.linspace(0.0, domain_end, count)
    fine_grid = np.linspace(0.0, domain_end, max(1000, count))
    names = experiment_config.get("ood_functions", ["linear", "sin_pi", "sin_2pi"])
    all_branch, all_trunk, all_target, all_labels = [], [], [], []
    for name in names:
        fine_values = analytic_input(str(name), fine_grid)
        branch = analytic_input(str(name), sensors)
        if experiment == "antiderivative":
            _require_scipy("antiderivative OOD reference")
            integral = cumulative_trapezoid(fine_values, fine_grid, initial=0.0)
            target = np.interp(queries, fine_grid, integral)
        elif experiment == "nonlinear_ode":
            target = solve_nonlinear_ode(
                fine_grid, fine_values, queries, config["solver_defaults"]
            )
        elif experiment == "pendulum":
            target = solve_pendulum(
                fine_grid,
                fine_values,
                queries,
                float(experiment_config["k"]),
                config["solver_defaults"],
            )
        else:
            raise ValueError(f"Unsupported experiment {experiment!r}")
        all_branch.append(np.repeat(branch[None, :], count, axis=0))
        all_trunk.append(queries[:, None])
        all_target.append(target[:, None])
        all_labels.extend([str(name)] * count)
    return {
        "branch": np.concatenate(all_branch).astype(np.float32),
        "trunk": np.concatenate(all_trunk).astype(np.float32),
        "target": np.concatenate(all_target).astype(np.float32),
        "labels": np.asarray(all_labels),
    }


def generate_pde_grid_case(config: Mapping[str, Any], seed: int) -> Dict[str, np.ndarray]:
    experiment_config = config["experiments"]["diffusion_reaction"]
    sampler = FunctionSpaceSampler(config["function_space"], 1.0)
    rng = np.random.default_rng(seed)
    fine_values = sampler.sample(1, rng)[0]
    sensors = np.linspace(0.0, 1.0, int(experiment_config["sensor_points"]))
    branch_vector = sampler.interpolate(fine_values, sensors).astype(np.float32)
    spatial_grid = np.linspace(0.0, 1.0, int(experiment_config["space_points"]))
    source = sampler.interpolate(fine_values, spatial_grid)
    x_grid, t_grid, field = solve_diffusion_reaction(
        source,
        diffusion=float(experiment_config["diffusion"]),
        reaction=float(experiment_config["reaction"]),
        space_points=int(experiment_config["space_points"]),
        time_points=int(experiment_config["time_points"]),
        solver_config=config["solver_defaults"],
    )
    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)
    trunk = np.column_stack((x_mesh.ravel(), t_mesh.ravel())).astype(np.float32)
    return {
        "branch": np.repeat(branch_vector[None, :], len(trunk), axis=0),
        "trunk": trunk,
        "target": field.reshape(-1, 1).astype(np.float32),
        "source": source.astype(np.float32),
        "x": x_grid.astype(np.float32),
        "t": t_grid.astype(np.float32),
        "grid_shape": np.asarray(field.shape, dtype=np.int64),
    }


__all__ = [
    "OperatorDataset",
    "FunctionSpaceSampler",
    "resolve_config",
    "build_split",
    "build_datasets",
    "generate_ood_data",
    "generate_pde_grid_case",
    "solve_antiderivative",
    "solve_nonlinear_ode",
    "solve_pendulum",
    "solve_diffusion_reaction",
]
