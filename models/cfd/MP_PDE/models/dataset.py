"""E3 trajectory generation and HDF5 loading for MP-PDE.

This is an independent implementation from the equations and numerical-method
description in arXiv:2202.03376.  No official repository source is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


SPLITS = ("train", "valid", "test")
PARAMETER_ORDER = ("alpha", "beta", "gamma")
PARALLEL_SCHEMA_VERSION = "mp_pde_e3_parallel_resume_v1"
_WORKER_X: Optional[np.ndarray] = None
_WORKER_TIMES: Optional[np.ndarray] = None
_WORKER_CONFIG: Optional[Dict[str, Any]] = None
_WORKER_STRIDE: Optional[int] = None


def _shift(values: np.ndarray, offset: int) -> np.ndarray:
    """Return values[i + offset] under periodic indexing."""
    return np.roll(values, -offset, axis=-1)


def _weno5_left(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Fifth-order WENO left state at every i+1/2 interface."""
    um2, um1, u0 = _shift(values, -2), _shift(values, -1), values
    up1, up2 = _shift(values, 1), _shift(values, 2)
    p0 = (2.0 * um2 - 7.0 * um1 + 11.0 * u0) / 6.0
    p1 = (-um1 + 5.0 * u0 + 2.0 * up1) / 6.0
    p2 = (2.0 * u0 + 5.0 * up1 - up2) / 6.0
    b0 = (13.0 / 12.0) * (um2 - 2.0 * um1 + u0) ** 2 + 0.25 * (um2 - 4.0 * um1 + 3.0 * u0) ** 2
    b1 = (13.0 / 12.0) * (um1 - 2.0 * u0 + up1) ** 2 + 0.25 * (um1 - up1) ** 2
    b2 = (13.0 / 12.0) * (u0 - 2.0 * up1 + up2) ** 2 + 0.25 * (3.0 * u0 - 4.0 * up1 + up2) ** 2
    alpha = np.stack((0.1 / (epsilon + b0) ** 2, 0.6 / (epsilon + b1) ** 2, 0.3 / (epsilon + b2) ** 2))
    weights = alpha / np.sum(alpha, axis=0, keepdims=True)
    return weights[0] * p0 + weights[1] * p1 + weights[2] * p2


def _weno5_right(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Fifth-order WENO right state at every i+1/2 interface."""
    um1, u0 = _shift(values, -1), values
    up1, up2, up3 = _shift(values, 1), _shift(values, 2), _shift(values, 3)
    p0 = (2.0 * up3 - 7.0 * up2 + 11.0 * up1) / 6.0
    p1 = (-up2 + 5.0 * up1 + 2.0 * u0) / 6.0
    p2 = (2.0 * up1 + 5.0 * u0 - um1) / 6.0
    b0 = (13.0 / 12.0) * (up1 - 2.0 * up2 + up3) ** 2 + 0.25 * (3.0 * up1 - 4.0 * up2 + up3) ** 2
    b1 = (13.0 / 12.0) * (u0 - 2.0 * up1 + up2) ** 2 + 0.25 * (u0 - up2) ** 2
    b2 = (13.0 / 12.0) * (um1 - 2.0 * u0 + up1) ** 2 + 0.25 * (um1 - 4.0 * u0 + 3.0 * up1) ** 2
    alpha = np.stack((0.1 / (epsilon + b0) ** 2, 0.6 / (epsilon + b1) ** 2, 0.3 / (epsilon + b2) ** 2))
    weights = alpha / np.sum(alpha, axis=0, keepdims=True)
    return weights[0] * p0 + weights[1] * p1 + weights[2] * p2


def godunov_quadratic_flux(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Godunov flux for the convex scalar flux f(u)=u**2."""
    left_flux, right_flux = left * left, right * right
    rarefaction = left <= right
    rare_flux = np.where((left <= 0.0) & (right >= 0.0), 0.0, np.minimum(left_flux, right_flux))
    shock_flux = np.maximum(left_flux, right_flux)
    return np.where(rarefaction, rare_flux, shock_flux)


def weno5_flux_derivative(values: np.ndarray, dx: float, epsilon: float = 1.0e-6) -> np.ndarray:
    """Conservative derivative d_x(u**2) on a periodic uniform grid."""
    interface_flux = godunov_quadratic_flux(_weno5_left(values, epsilon), _weno5_right(values, epsilon))
    return (interface_flux - np.roll(interface_flux, 1, axis=-1)) / dx


def fourth_order_second_derivative(values: np.ndarray, dx: float) -> np.ndarray:
    return (-_shift(values, 2) + 16.0 * _shift(values, 1) - 30.0 * values + 16.0 * _shift(values, -1) - _shift(values, -2)) / (12.0 * dx**2)


def fourth_order_third_derivative(values: np.ndarray, dx: float) -> np.ndarray:
    return (_shift(values, -3) - 8.0 * _shift(values, -2) + 13.0 * _shift(values, -1) - 13.0 * _shift(values, 1) + 8.0 * _shift(values, 2) - _shift(values, 3)) / (8.0 * dx**3)


def sample_e3_parameters(rng: np.random.Generator, cfg: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    equation, forcing = cfg["equation"], cfg["forcing"]
    params = np.array(
        [rng.uniform(*equation["alpha_range"]), rng.uniform(*equation["beta_range"]), rng.uniform(*equation["gamma_range"])],
        dtype=np.float64,
    )
    policy = cfg["ambiguity_policy"]
    if policy == "paper_strict":
        omega_range = forcing["paper_omega_range"]
    elif policy == "official_consistency":
        omega_range = forcing["official_consistency_omega_range"]
    else:
        raise ValueError(f"Unknown ambiguity_policy={policy!r}")
    terms = int(forcing["terms"])
    provenance = {
        "amplitude": rng.uniform(*forcing["amplitude_range"], size=terms),
        "omega": rng.uniform(*omega_range, size=terms),
        "mode": rng.choice(np.asarray(forcing["modes"], dtype=np.int64), size=terms),
        "phase": rng.uniform(*forcing["phase_range"], size=terms),
    }
    return params, provenance


def evaluate_forcing(time: float, x: np.ndarray, forcing: Mapping[str, np.ndarray], domain_length: float) -> np.ndarray:
    phase = (
        forcing["omega"][:, None] * time
        + 2.0 * np.pi * forcing["mode"][:, None] * x[None, :] / domain_length
        + forcing["phase"][:, None]
    )
    return np.sum(forcing["amplitude"][:, None] * np.sin(phase), axis=0)


def e3_rhs(
    time: float,
    state: np.ndarray,
    x: np.ndarray,
    params: np.ndarray,
    forcing: Mapping[str, np.ndarray],
    domain_length: float,
    weno_epsilon: float,
) -> np.ndarray:
    alpha, beta, gamma = params
    dx = domain_length / state.shape[-1]
    return (
        evaluate_forcing(time, x, forcing, domain_length)
        - alpha * weno5_flux_derivative(state, dx, weno_epsilon)
        + beta * fourth_order_second_derivative(state, dx)
        - gamma * fourth_order_third_derivative(state, dx)
    )


def _stable_step(state: np.ndarray, params: np.ndarray, dx: float, cfl: float) -> float:
    alpha, beta, gamma = np.abs(params)
    limits = []
    wave_speed = 2.0 * alpha * float(np.max(np.abs(state)))
    if wave_speed > 1.0e-14:
        limits.append(dx / wave_speed)
    if beta > 1.0e-14:
        limits.append(dx**2 / (2.0 * beta))
    if gamma > 1.0e-14:
        limits.append(dx**3 / (6.0 * gamma))
    return cfl * min(limits) if limits else np.inf


def generate_trajectory(
    x: np.ndarray,
    save_times: np.ndarray,
    params: np.ndarray,
    forcing: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """Integrate one trajectory using RK4 and stability-limited substeps."""
    domain_length = float(cfg["domain_length"])
    generation = cfg["generation"]
    dx = domain_length / x.size
    state = evaluate_forcing(float(save_times[0]), x, forcing, domain_length).astype(np.float64)
    trajectory = np.empty((save_times.size, x.size), dtype=np.float32)
    trajectory[0] = state
    current_time = float(save_times[0])
    for output_index, target_time in enumerate(save_times[1:], start=1):
        substeps = 0
        while current_time < float(target_time) - 1.0e-14:
            stable = _stable_step(state, params, dx, float(generation["cfl"]))
            remaining = float(target_time) - current_time
            step = min(stable, remaining)
            if step < float(generation["min_dt"]) and remaining > float(generation["min_dt"]):
                raise RuntimeError(f"Stable RK4 step {step:.3e} fell below min_dt at t={current_time:.6g}")
            step = remaining if remaining <= float(generation["min_dt"]) else step
            rhs_args = (x, params, forcing, domain_length, float(generation["weno_epsilon"]))
            k1 = e3_rhs(current_time, state, *rhs_args)
            k2 = e3_rhs(current_time + 0.5 * step, state + 0.5 * step * k1, *rhs_args)
            k3 = e3_rhs(current_time + 0.5 * step, state + 0.5 * step * k2, *rhs_args)
            k4 = e3_rhs(current_time + step, state + step * k3, *rhs_args)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current_time += step
            substeps += 1
            if substeps > int(generation["max_substeps"]):
                raise RuntimeError(f"max_substeps exceeded while advancing to t={target_time:.6g}")
            if not np.all(np.isfinite(state)):
                raise FloatingPointError(f"Non-finite E3 state at t={current_time:.6g}")
        current_time = float(target_time)
        trajectory[output_index] = state
    return trajectory


def _generation_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    data = config["data"]
    return {
        "ambiguity_policy": config["experiment"]["ambiguity_policy"],
        "domain_length": data["domain_length"],
        "equation": data["equation"],
        "forcing": data["forcing"],
        "generation": data["generation"],
    }


def _initialize_generation_worker(
    x_high: np.ndarray, times: np.ndarray, generation_cfg: Mapping[str, Any], stride: int
) -> None:
    """Initialize immutable state used by one trajectory worker process."""
    global _WORKER_X, _WORKER_TIMES, _WORKER_CONFIG, _WORKER_STRIDE
    _WORKER_X = x_high
    _WORKER_TIMES = times
    _WORKER_CONFIG = dict(generation_cfg)
    _WORKER_STRIDE = int(stride)


def _generate_sample_worker(payload: Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[int, np.ndarray]:
    """Generate one independent trajectory; HDF5 remains owned by the parent."""
    if _WORKER_X is None or _WORKER_TIMES is None or _WORKER_CONFIG is None or _WORKER_STRIDE is None:
        raise RuntimeError("Parallel E3 worker was not initialized")
    sample_index, params, amplitude, omega, mode, phase = payload
    forcing = {"amplitude": amplitude, "omega": omega, "mode": mode, "phase": phase}
    trajectory = generate_trajectory(_WORKER_X, _WORKER_TIMES, params, forcing, _WORKER_CONFIG)
    return sample_index, trajectory[:, ::_WORKER_STRIDE]


def _sample_payloads(group: h5py.Group, indices: Iterable[int]) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    for sample_index in indices:
        yield (
            sample_index,
            np.asarray(group["params"][sample_index], dtype=np.float64),
            np.asarray(group["forcing_amplitude"][sample_index], dtype=np.float64),
            np.asarray(group["forcing_omega"][sample_index], dtype=np.float64),
            np.asarray(group["forcing_mode"][sample_index], dtype=np.int64),
            np.asarray(group["forcing_phase"][sample_index], dtype=np.float64),
        )


def _parallel_results(
    payloads: Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    workers: int,
    max_in_flight: int,
    x_high: np.ndarray,
    times: np.ndarray,
    generation_cfg: Mapping[str, Any],
    stride: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    """Yield completed trajectories while keeping the process queue bounded."""
    if workers == 1:
        _initialize_generation_worker(x_high, times, generation_cfg, stride)
        for payload in payloads:
            yield _generate_sample_worker(payload)
        return

    thread_variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous_environment = {name: os.environ.get(name) for name in thread_variables}
    for name in thread_variables:
        os.environ[name] = "1"
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_generation_worker,
        initargs=(x_high, times, dict(generation_cfg), stride),
    )
    iterator = iter(payloads)
    futures = set()
    try:
        for _ in range(max_in_flight):
            try:
                futures.add(executor.submit(_generate_sample_worker, next(iterator)))
            except StopIteration:
                break
        while futures:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    futures.add(executor.submit(_generate_sample_worker, next(iterator)))
                except StopIteration:
                    pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _generation_signature(
    config: Mapping[str, Any], counts: Mapping[str, int], nt: int, high_nx: int, target_nx: int
) -> str:
    payload = {
        "schema_version": PARALLEL_SCHEMA_VERSION,
        "paper": str(config["experiment"]["paper"]),
        "ambiguity_policy": str(config["experiment"]["ambiguity_policy"]),
        "generation_config": _generation_config(config),
        "counts": {name: int(counts[name]) for name in SPLITS},
        "nt": int(nt),
        "high_resolution_nx": int(high_nx),
        "resolution": int(target_nx),
        "seed": int(config["data"]["seed"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _initialize_partial_file(
    partial_path: Path,
    config: Mapping[str, Any],
    counts: Mapping[str, int],
    nt: int,
    high_nx: int,
    target_nx: int,
    x: np.ndarray,
    times: np.ndarray,
    generation_cfg: Mapping[str, Any],
    compression: Optional[str],
    compression_opts: Optional[int],
    signature: str,
) -> None:
    data = config["data"]
    base_seed = int(data["seed"])
    with h5py.File(partial_path, "w") as handle:
        handle.attrs["schema_version"] = PARALLEL_SCHEMA_VERSION
        handle.attrs["generation_signature"] = signature
        handle.attrs["experiment"] = "MP-PDE E3"
        handle.attrs["paper"] = str(config["experiment"]["paper"])
        handle.attrs["ambiguity_policy"] = str(config["experiment"]["ambiguity_policy"])
        handle.attrs["parameter_order"] = json.dumps(PARAMETER_ORDER)
        handle.attrs["generation_config"] = json.dumps(generation_cfg, sort_keys=True)
        handle.attrs["seed"] = base_seed
        handle.attrs["high_resolution_nx"] = high_nx
        handle.attrs["resolution"] = target_nx
        for split_index, split in enumerate(SPLITS):
            count = int(counts[split])
            rng = np.random.default_rng(np.random.SeedSequence([base_seed, split_index]))
            group = handle.create_group(split)
            group.attrs["seed_derivation"] = json.dumps([base_seed, split_index])
            group.create_dataset("x", data=x)
            group.create_dataset("t", data=times.astype(np.float32))
            group.create_dataset(
                "u", shape=(count, nt, target_nx), dtype="f4", chunks=(1, min(nt, 32), target_nx),
                compression=compression, compression_opts=compression_opts,
            )
            params_ds = group.create_dataset("params", shape=(count, 3), dtype="f4")
            terms = int(data["forcing"]["terms"])
            amplitude_ds = group.create_dataset("forcing_amplitude", shape=(count, terms), dtype="f4")
            omega_ds = group.create_dataset("forcing_omega", shape=(count, terms), dtype="f4")
            mode_ds = group.create_dataset("forcing_mode", shape=(count, terms), dtype="i8")
            phase_ds = group.create_dataset("forcing_phase", shape=(count, terms), dtype="f4")
            group.create_dataset("completed", shape=(count,), dtype="bool", data=np.zeros(count, dtype=bool))
            for sample_index in range(count):
                params, forcing = sample_e3_parameters(rng, generation_cfg)
                params_ds[sample_index] = params
                amplitude_ds[sample_index] = forcing["amplitude"]
                omega_ds[sample_index] = forcing["omega"]
                mode_ds[sample_index] = forcing["mode"]
                phase_ds[sample_index] = forcing["phase"]
        handle.flush()


def generate_e3_hdf5(
    config: Mapping[str, Any],
    output_path: Path | str,
    *,
    sample_counts: Optional[Mapping[str, int]] = None,
    nt: Optional[int] = None,
    high_resolution_nx: Optional[int] = None,
    resolution: Optional[int] = None,
    workers: Optional[int] = None,
    max_in_flight: Optional[int] = None,
    flush_every: Optional[int] = None,
    resume_partial: Optional[bool] = None,
    overwrite: bool = False,
) -> Path:
    """Generate E3 splits with process workers and a single resumable HDF5 writer."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    data = config["data"]
    parallel = data.get("parallel_generation", {})
    workers = int(workers if workers is not None else parallel.get("workers", 1))
    max_in_flight = int(max_in_flight if max_in_flight is not None else parallel.get("max_in_flight", 2 * workers))
    flush_every = int(flush_every if flush_every is not None else parallel.get("flush_every", workers))
    resume_partial = bool(resume_partial if resume_partial is not None else parallel.get("resume_partial", True))
    if workers < 1 or max_in_flight < workers or flush_every < 1:
        raise ValueError("workers>=1, max_in_flight>=workers, and flush_every>=1 are required")
    nt = int(nt or data["num_time_points"])
    high_nx = int(high_resolution_nx or data["high_resolution_nx"])
    target_nx = int(resolution or data["resolution"])
    if high_nx % target_nx != 0:
        raise ValueError(f"high_resolution_nx={high_nx} must be divisible by resolution={target_nx}")
    if high_nx < 7 or nt < 2:
        raise ValueError("Generation requires high_resolution_nx>=7 and nt>=2")
    counts = dict(sample_counts or {name: int(data[f"{name}_samples"]) for name in SPLITS})
    if set(counts) != set(SPLITS) or any(int(counts[name]) <= 0 for name in SPLITS):
        raise ValueError(f"sample_counts must provide positive counts for {SPLITS}")

    domain_length, final_time = float(data["domain_length"]), float(data["final_time"])
    x_high = np.linspace(0.0, domain_length, high_nx, endpoint=False, dtype=np.float64)
    stride = high_nx // target_nx
    x = x_high[::stride].astype(np.float32)
    times = np.linspace(0.0, final_time, nt, dtype=np.float64)
    generation_cfg = _generation_config(config)
    compression = data["generation"].get("compression")
    compression_opts = int(data["generation"].get("compression_level", 4)) if compression == "gzip" else None
    signature = _generation_signature(config, counts, nt, high_nx, target_nx)
    if partial_path.exists() and overwrite:
        partial_path.unlink()
    if partial_path.exists() and not resume_partial:
        raise FileExistsError(f"Partial dataset exists; enable resume_partial or use --overwrite: {partial_path}")
    if not partial_path.exists():
        _initialize_partial_file(
            partial_path, config, counts, nt, high_nx, target_nx, x, times, generation_cfg,
            compression, compression_opts, signature,
        )

    try:
        with h5py.File(partial_path, "r+") as handle:
            stored_signature = str(handle.attrs.get("generation_signature", ""))
            if stored_signature != signature:
                raise ValueError("Partial dataset configuration does not match this run; archive it or use --overwrite")
            print(
                f"[generate] workers={workers} max_in_flight={max_in_flight} flush_every={flush_every} "
                f"resume_partial={resume_partial}", flush=True,
            )
            for split in SPLITS:
                count = int(counts[split])
                group = handle[split]
                completed_ds = group["completed"]
                pending_indices = np.flatnonzero(~np.asarray(completed_ds[:], dtype=bool)).tolist()
                completed_count = count - len(pending_indices)
                if pending_indices:
                    print(f"[generate] split={split} resume_completed={completed_count}/{count}", flush=True)
                payloads = _sample_payloads(group, pending_indices)
                for sample_index, trajectory in _parallel_results(
                    payloads, workers, max_in_flight, x_high, times, generation_cfg, stride
                ):
                    group["u"][sample_index] = trajectory
                    completed_ds[sample_index] = True
                    completed_count += 1
                    if completed_count % flush_every == 0 or completed_count == count:
                        handle.flush()
                    print(
                        f"[generate] split={split} completed={completed_count}/{count} "
                        f"sample_index={sample_index} workers={workers}", flush=True,
                    )
                if not np.all(np.asarray(completed_ds[:], dtype=bool)):
                    raise RuntimeError(f"Split {split} is incomplete after generation")
                handle.flush()
        os.replace(partial_path, output_path)
    except Exception:
        print(f"Generation failed; partial file retained at {partial_path}", flush=True)
        raise
    return output_path


class E3Dataset(Dataset):
    """Lazy, process-safe reader for one E3 HDF5 split."""

    def __init__(self, path: Path | str, split: str, expected_nt: Optional[int] = None, expected_nx: Optional[int] = None):
        self.path = Path(path)
        self.split = split
        self._handle: Optional[h5py.File] = None
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if not self.path.is_file():
            raise FileNotFoundError(f"E3 dataset not found: {self.path}")
        with h5py.File(self.path, "r") as handle:
            if split not in handle:
                raise KeyError(f"Missing HDF5 group {split!r}")
            group = handle[split]
            required = {"u", "x", "t", "params", "forcing_amplitude", "forcing_omega", "forcing_mode", "forcing_phase"}
            missing = required.difference(group.keys())
            if missing:
                raise KeyError(f"Missing HDF5 fields in {split}: {sorted(missing)}")
            shape = group["u"].shape
            if len(shape) != 3 or group["params"].shape != (shape[0], 3) or group["x"].shape != (shape[2],) or group["t"].shape != (shape[1],):
                raise ValueError(f"Inconsistent E3 schema in split={split}: u={shape}")
            if expected_nt is not None and shape[1] != expected_nt:
                raise ValueError(f"Expected nt={expected_nt}, found {shape[1]}")
            if expected_nx is not None and shape[2] != expected_nx:
                raise ValueError(f"Expected nx={expected_nx}, found {shape[2]}")
            self.length, self.nt, self.nx = shape
            x, t = group["x"][:], group["t"][:]
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
                raise ValueError("Grid/time metadata are non-finite or non-monotone")
            if x.size > 1 and not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-5, atol=1e-7):
                raise ValueError("E3 x grid must be uniform")
            if t.size > 1 and not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-5, atol=1e-7):
                raise ValueError("E3 saved times must be uniform")

    def _group(self) -> h5py.Group:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[self.split]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        group = self._group()
        trajectory = np.asarray(group["u"][index], dtype=np.float32)
        params = np.asarray(group["params"][index], dtype=np.float32)
        if not np.all(np.isfinite(trajectory)) or not np.all(np.isfinite(params)):
            raise FloatingPointError(f"Non-finite data at split={self.split}, sample={index}")
        return {
            "u": torch.from_numpy(trajectory),
            "x": torch.from_numpy(np.asarray(group["x"][:], dtype=np.float32)),
            "t": torch.from_numpy(np.asarray(group["t"][:], dtype=np.float32)),
            "params": torch.from_numpy(params),
            "index": torch.tensor(index, dtype=torch.long),
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate the MP-PDE E3 HDF5 dataset")
    parser.add_argument("--config", type=Path, default=project_root / "config/config.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--valid-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--high-resolution-nx", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--flush-every", type=int)
    parser.add_argument("--no-resume-partial", action="store_true")
    args = parser.parse_args()
    config = _load_yaml(args.config.resolve())
    configured = Path(config["paths"]["data"])
    output = args.output or (configured if configured.is_absolute() else project_root / configured)
    default_counts = {name: int(config["data"][f"{name}_samples"]) for name in SPLITS}
    counts = {
        "train": args.train_samples or default_counts["train"],
        "valid": args.valid_samples or default_counts["valid"],
        "test": args.test_samples or default_counts["test"],
    }
    generated = generate_e3_hdf5(
        config, output, sample_counts=counts, nt=args.nt, high_resolution_nx=args.high_resolution_nx,
        resolution=args.resolution, workers=args.workers, max_in_flight=args.max_in_flight,
        flush_every=args.flush_every, resume_partial=False if args.no_resume_partial else None,
        overwrite=args.overwrite,
    )
    print(f"Generated E3 dataset: {generated}", flush=True)


if __name__ == "__main__":
    main()
