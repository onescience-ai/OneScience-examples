#!/usr/bin/env python3
"""Run non-interactive GraphCast inference and save verified artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = Path(os.environ.get("GRAPHCAST_SOURCE_DIR", SCRIPT_DIR)).resolve()
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout


LOGGER = logging.getLogger("graphcast_inference")


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def find_file(assets_dir: Path, value: str | None, kind: str, default_pattern: str) -> Path:
    if value:
        candidate = Path(value)
        if candidate.is_file():
            return candidate.resolve()
        direct = assets_dir / candidate
        if direct.is_file():
            return direct.resolve()
        matches = list(assets_dir.rglob(value))
    else:
        matches = list(assets_dir.rglob(default_pattern))
    matches = [path for path in matches if path.is_file() and not path.name.endswith(".part")]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {kind} under {assets_dir}; found {len(matches)}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0].resolve()


def require_stats(assets_dir: Path) -> tuple[Path, Path, Path]:
    names = (
        "diffs_stddev_by_level.nc",
        "mean_by_level.nc",
        "stddev_by_level.nc",
    )
    paths = []
    for name in names:
        matches = [path for path in assets_dir.rglob(name) if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {name} under {assets_dir}, found {len(matches)}")
        paths.append(matches[0].resolve())
    return tuple(paths)  # type: ignore[return-value]


def dims_dict(dataset: xarray.Dataset) -> dict[str, int]:
    return {name: int(size) for name, size in dataset.sizes.items()}


def materialize(dataset: xarray.Dataset) -> xarray.Dataset:
    values = {
        name: np.asarray(variable.data)
        for name, variable in dataset.data_vars.items()
    }
    return dataset.copy(data=values)


def validate_and_measure(predictions: xarray.Dataset, targets: xarray.Dataset) -> dict[str, Any]:
    per_variable: dict[str, Any] = {}
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    total_nonfinite = 0
    for name in predictions.data_vars:
        prediction = np.asarray(predictions[name].values)
        finite = np.isfinite(prediction)
        entry: dict[str, Any] = {
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "finite_count": int(finite.sum()),
            "nan_count": int(np.isnan(prediction).sum()),
            "inf_count": int(np.isinf(prediction).sum()),
            "min": float(np.nanmin(prediction)) if finite.any() else None,
            "max": float(np.nanmax(prediction)) if finite.any() else None,
        }
        total_nonfinite += int((~finite).sum())
        if name in targets:
            target = np.asarray(targets[name].values)
            valid = finite & np.isfinite(target)
            if valid.any():
                difference = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
                absolute_sum = float(np.abs(difference).sum(dtype=np.float64))
                squared_sum = float(np.square(difference).sum(dtype=np.float64))
                count = int(difference.size)
                entry["mae"] = absolute_sum / count
                entry["rmse"] = math.sqrt(squared_sum / count)
                entry["metric_count"] = count
                total_abs += absolute_sum
                total_sq += squared_sum
                total_count += count
            else:
                entry["metric_unavailable_reason"] = "No colocated finite target/prediction values"
        else:
            entry["metric_unavailable_reason"] = "Target variable is absent"
        per_variable[name] = entry
    return {
        "per_variable": per_variable,
        "aggregate": {
            "mae": total_abs / total_count if total_count else None,
            "rmse": math.sqrt(total_sq / total_count) if total_count else None,
            "count": total_count,
            "note": "Element-weighted aggregate across variables; units are mixed, so per-variable metrics are preferred.",
        },
        "nonfinite_prediction_count": total_nonfinite,
    }


def select_2d(data: xarray.DataArray) -> xarray.DataArray:
    selected = data
    for dim in list(selected.dims):
        if dim not in ("lat", "lon"):
            if dim == "level" and 500 in selected.coords.get("level", []):
                selected = selected.sel(level=500)
            else:
                selected = selected.isel({dim: 0})
    return selected.squeeze(drop=True)


def save_plot(predictions: xarray.Dataset, targets: xarray.Dataset, output_dir: Path) -> Path | None:
    candidates = ["2m_temperature", "mean_sea_level_pressure"]
    name = next((item for item in candidates if item in predictions and item in targets), None)
    if name is None:
        LOGGER.warning("No supported common variable found for the sample plot")
        return None
    pred = np.asarray(select_2d(predictions[name]).values)
    target = np.asarray(select_2d(targets[name]).values)
    difference = pred - target
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, values, title, cmap in (
        (axes[0], target, f"Target: {name}", "viridis"),
        (axes[1], pred, f"Prediction: {name}", "viridis"),
        (axes[2], difference, "Prediction - target", "RdBu_r"),
    ):
        image = axis.imshow(values, origin="lower", cmap=cmap, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("longitude index")
        axis.set_ylabel("latitude index")
        figure.colorbar(image, ax=axis, shrink=0.8)
    path = output_dir / f"prediction_{name}.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def build_runner(
    model_config,
    task_config,
    params,
    state,
    diffs_stddev_by_level,
    mean_by_level,
    stddev_by_level,
):
    def construct_wrapped_graphcast():
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        return autoregressive.Predictor(predictor, gradient_checkpointing=True)

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return construct_wrapped_graphcast()(
            inputs, targets_template=targets_template, forcings=forcings
        )

    transformed = jax.jit(run_forward.apply)

    def apply(*, rng, inputs, targets_template, forcings):
        predictions, _ = transformed(
            params, state, rng, inputs, targets_template, forcings
        )
        return predictions

    return apply


def parse_args() -> argparse.Namespace:
    root = Path(os.environ.get("WEATHER_DATA_ROOT", "/root/group_data/SDU-Test/weather"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=root / "assets")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--model", default="GraphCast_small", help="Checkpoint path, filename, or model name")
    parser.add_argument("--dataset", help="Dataset path or filename; defaults to compatible 1-step ERA5 sample")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--log-file", type=Path, default=root / "logs" / "run_inference.log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file
    configure_logging(log_file)
    started = datetime.now(timezone.utc)
    LOGGER.info("Starting GraphCast inference")
    LOGGER.info("Source directory: %s", SOURCE_DIR)
    LOGGER.info("JAX backend: %s", jax.default_backend())
    LOGGER.info("JAX devices: %s", [str(device) for device in jax.devices()])
    LOGGER.info("FlagGems: not enabled; this is a JAX/Haiku execution path")

    model_pattern = "*GraphCast_small*.npz" if args.model == "GraphCast_small" else args.model
    model_file = find_file(args.assets_dir, None if args.model == "GraphCast_small" else args.model, "checkpoint", model_pattern)
    dataset_file = find_file(
        args.assets_dir,
        args.dataset,
        "dataset",
        "source-era5*_res-1.0_levels-13_steps-01.nc",
    )
    diffs_path, mean_path, stddev_path = require_stats(args.assets_dir)
    LOGGER.info("Checkpoint: %s", model_file)
    LOGGER.info("Dataset: %s", dataset_file)

    with model_file.open("rb") as stream:
        ckpt = checkpoint.load(stream, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    LOGGER.info("Model description: %s", ckpt.description)
    LOGGER.info("Model license: %s", ckpt.license)

    with xarray.open_dataset(dataset_file) as opened:
        example_batch = opened.load()
    if example_batch.sizes.get("time", 0) < args.steps + 2:
        raise ValueError(
            f"Dataset has {example_batch.sizes.get('time', 0)} time entries; "
            f"{args.steps} forecast steps require at least {args.steps + 2}"
        )
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{args.steps * 6}h"),
        **dataclasses.asdict(task_config),
    )
    LOGGER.info("Input dimensions: %s", dims_dict(inputs))
    LOGGER.info("Target dimensions: %s", dims_dict(targets))
    LOGGER.info("Forcing dimensions: %s", dims_dict(forcings))
    if model_config.resolution not in (0, 360.0 / inputs.sizes["lon"]):
        raise ValueError("Model resolution does not match the selected dataset")

    with xarray.open_dataset(diffs_path) as opened:
        diffs = opened.load()
    with xarray.open_dataset(mean_path) as opened:
        mean = opened.load()
    with xarray.open_dataset(stddev_path) as opened:
        stddev = opened.load()

    runner = build_runner(model_config, task_config, params, state, diffs, mean, stddev)

    def predict() -> xarray.Dataset:
        result = rollout.chunked_prediction(
            runner,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets * np.nan,
            forcings=forcings,
        )
        return materialize(result)

    first_start = time.perf_counter()
    predictions = predict()
    first_seconds = time.perf_counter() - first_start
    LOGGER.info("First JIT compile + inference: %.6f seconds", first_seconds)

    warm_start = time.perf_counter()
    warm_predictions = predict()
    warm_seconds = time.perf_counter() - warm_start
    LOGGER.info("Second warm inference: %.6f seconds", warm_seconds)
    del warm_predictions

    output_netcdf = args.output_dir / "predictions.nc"
    predictions.to_netcdf(output_netcdf)
    with xarray.open_dataset(output_netcdf) as reopened:
        reopened.load()
        reread_ok = True
        reread_dims = dims_dict(reopened)

    validation = validate_and_measure(predictions, targets)
    plot_path = save_plot(predictions, targets, args.output_dir)
    finished = datetime.now(timezone.utc)
    metrics = {
        "status": "success",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "execution_label": "CPU baseline" if jax.default_backend() == "cpu" else "accelerator execution",
        "flag_gems": {
            "used": False,
            "reason": "GraphCast is implemented in JAX/Haiku; no supported FlagGems execution path was confirmed.",
        },
        "timing_seconds": {
            "first_jit_compile_and_inference": first_seconds,
            "second_warm_inference": warm_seconds,
        },
        "forecast_steps": args.steps,
        "forecast_hours": args.steps * 6,
        "files": {
            "checkpoint": str(model_file),
            "dataset": str(dataset_file),
            "diffs_stddev": str(diffs_path),
            "mean": str(mean_path),
            "stddev": str(stddev_path),
            "predictions_netcdf": str(output_netcdf),
            "sample_plot": str(plot_path) if plot_path else None,
            "run_log": str(log_file),
        },
        "dimensions": {
            "inputs": dims_dict(inputs),
            "targets": dims_dict(targets),
            "forcings": dims_dict(forcings),
            "predictions": dims_dict(predictions),
            "reread_predictions": reread_dims,
        },
        "prediction_variables": sorted(predictions.data_vars),
        "validation": {
            "netcdf_reread_ok": reread_ok,
            **validation,
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("NetCDF reread: %s", reread_ok)
    LOGGER.info("Non-finite prediction values: %d", validation["nonfinite_prediction_count"])
    LOGGER.info("Metrics: %s", metrics_path)
    LOGGER.info("Predictions: %s", output_netcdf)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        if LOGGER.handlers:
            LOGGER.exception("Inference failed")
        else:
            logging.basicConfig(level=logging.ERROR)
            logging.exception("Inference failed before logging was configured")
        raise
