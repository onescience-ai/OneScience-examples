#!/usr/bin/env python3
"""Validate real FNO outputs and render paper-comparison figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import compute_metrics  # noqa: E402
from train import atomic_write_json, load_config, resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate FNO inference artifacts and generate scientific figures."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--sample-index", type=int, default=0, help="Local test-set index to visualize."
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON mapping in {path}")
    return payload


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_figure(figure: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)
    if path.stat().st_size == 0:
        raise RuntimeError(f"Generated an empty figure: {path}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary, path)


def verify_csv(
    path: Path,
    sample_indices: np.ndarray,
    full_metrics: np.ndarray,
    lead_metrics: np.ndarray,
    time_values: np.ndarray,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Per-sample metrics CSV is missing: {path}")
    expected_header = ["sample_index", "relative_l2_full"] + [
        f"relative_l2_t{int(value)}" for value in time_values
    ]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != expected_header:
        raise ValueError(f"Unexpected CSV header in {path}: {rows[0] if rows else None}")
    if len(rows) - 1 != len(sample_indices):
        raise ValueError(f"Expected {len(sample_indices)} CSV rows, found {len(rows)-1}")
    for row_index, row in enumerate(rows[1:]):
        if int(row[0]) != int(sample_indices[row_index]):
            raise ValueError(f"CSV sample order mismatch at row {row_index + 2}")
        observed = np.asarray([float(value) for value in row[1:]], dtype=np.float64)
        expected = np.concatenate(
            ([full_metrics[row_index]], lead_metrics[row_index].astype(np.float64))
        )
        if not np.allclose(observed, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"CSV metric mismatch for sample {sample_indices[row_index]}")


def validate_history(history_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = history_payload.get("history")
    if not isinstance(records, list) or not records:
        raise ValueError("Training history contains no epoch records")
    formal = history_payload.get("run_type") == "formal"
    requested = int(history_payload.get("epochs_requested", len(records)))
    if formal and (requested != 500 or len(records) != 500):
        raise ValueError(
            f"Formal paper reproduction requires 500 epochs, got requested={requested}, "
            f"records={len(records)}"
        )
    required = (
        "epoch",
        "learning_rate",
        "duration_seconds",
        "train_step_loss_sum",
        "train_mean_step_relative_l2",
        "train_full_relative_l2",
        "test_mean_step_relative_l2",
        "test_full_relative_l2",
        "best",
    )
    for position, record in enumerate(records, start=1):
        missing = [key for key in required if key not in record]
        if missing:
            raise KeyError(f"Epoch record {position} is missing {missing}")
        if int(record["epoch"]) != position:
            raise ValueError(f"Epoch sequence is not contiguous at record {position}")
        numeric = [float(record[key]) for key in required[1:-1]]
        if not np.isfinite(numeric).all():
            raise FloatingPointError(f"Non-finite training history at epoch {position}")
    return records


def make_training_figure(
    records: list[dict[str, Any]], paper_metric: float, best_epoch: int
) -> plt.Figure:
    epochs = np.asarray([record["epoch"] for record in records], dtype=np.int64)
    train_full = np.asarray(
        [record["train_full_relative_l2"] for record in records], dtype=np.float64
    )
    test_full = np.asarray(
        [record["test_full_relative_l2"] for record in records], dtype=np.float64
    )
    train_loss = np.asarray(
        [record["train_step_loss_sum"] for record in records], dtype=np.float64
    )
    train_step = np.asarray(
        [record["train_mean_step_relative_l2"] for record in records], dtype=np.float64
    )
    test_step = np.asarray(
        [record["test_mean_step_relative_l2"] for record in records], dtype=np.float64
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    left = axes[0]
    left.plot(epochs, train_full, label="Train full relative L2", linewidth=1.6)
    left.plot(epochs, test_full, label="Test full relative L2", linewidth=1.6)
    left.axhline(
        paper_metric,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Paper benchmark ({paper_metric:.4f})",
    )
    left.axvline(
        best_epoch,
        color="tab:green",
        linestyle=":",
        linewidth=1.2,
        label=f"Best checkpoint epoch ({best_epoch})",
    )
    if np.all(train_full > 0) and np.all(test_full > 0):
        left.set_yscale("log")
    left.set_xlabel("Epoch")
    left.set_ylabel("Full-trajectory relative L2")
    left.set_title("FNO-2D rollout error")
    left.grid(True, alpha=0.25)
    left.legend(fontsize=8)

    right = axes[1]
    loss_line = right.plot(
        epochs,
        train_loss,
        color="tab:blue",
        label="Train 10-step loss sum",
        linewidth=1.5,
    )
    right.set_xlabel("Epoch")
    right.set_ylabel("Summed step relative L2", color="tab:blue")
    right.tick_params(axis="y", labelcolor="tab:blue")
    right.grid(True, alpha=0.25)
    diagnostic = right.twinx()
    train_line = diagnostic.plot(
        epochs,
        train_step,
        color="tab:orange",
        label="Train mean-step relative L2",
        linewidth=1.3,
    )
    test_line = diagnostic.plot(
        epochs,
        test_step,
        color="tab:red",
        label="Test mean-step relative L2",
        linewidth=1.3,
    )
    diagnostic.set_ylabel("Mean-step relative L2")
    right.set_title("Training objective and step diagnostics")
    lines = loss_line + train_line + test_line
    right.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
    return figure


def representative_leads(number_of_steps: int) -> list[int]:
    if number_of_steps <= 0:
        raise ValueError("At least one rollout step is required")
    return sorted({0, number_of_steps // 2, number_of_steps - 1})


def make_rollout_figure(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_indices: np.ndarray,
    time_values: np.ndarray,
    lead_metrics: np.ndarray,
    local_sample: int,
) -> plt.Figure:
    if not 0 <= local_sample < prediction.shape[0]:
        raise IndexError(
            f"sample-index {local_sample} is outside [0,{prediction.shape[0] - 1}]"
        )
    lead_indices = representative_leads(prediction.shape[-1])
    figure, axes = plt.subplots(
        len(lead_indices),
        3,
        figsize=(11.5, 3.25 * len(lead_indices)),
        squeeze=False,
        constrained_layout=True,
    )
    global_sample = int(sample_indices[local_sample])
    for row, lead_index in enumerate(lead_indices):
        truth = target[local_sample, :, :, lead_index]
        estimate = prediction[local_sample, :, :, lead_index]
        absolute_error = np.abs(estimate - truth)
        shared_limit = max(float(np.max(np.abs(truth))), float(np.max(np.abs(estimate))), 1e-12)
        error_limit = max(float(np.max(absolute_error)), 1e-12)
        time_value = int(time_values[lead_index])
        relative_error = float(lead_metrics[local_sample, lead_index])

        fields = (truth, estimate, absolute_error)
        titles = (
            f"Target vorticity w\nsample={global_sample}, t={time_value}",
            f"Predicted vorticity w\nrelative L2={relative_error:.5f}",
            f"Absolute error |prediction-target|\nt={time_value}",
        )
        for column, (field, title) in enumerate(zip(fields, titles)):
            axis = axes[row, column]
            if column < 2:
                image = axis.imshow(
                    field,
                    origin="lower",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    interpolation="nearest",
                    cmap="RdBu_r",
                    vmin=-shared_limit,
                    vmax=shared_limit,
                )
                color_label = "Vorticity w (unit not specified)"
            else:
                image = axis.imshow(
                    field,
                    origin="lower",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    interpolation="nearest",
                    cmap="magma",
                    vmin=0.0,
                    vmax=error_limit,
                )
                color_label = "Absolute error"
            axis.set_aspect("equal")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_title(title, fontsize=9)
            colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
            colorbar.set_label(color_label, fontsize=8)
    return figure


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = (
        resolve_project_path(config["paths"]["results_dir"])
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    if args.output_dir is None:
        history_path = resolve_project_path(config["paths"]["train_history"])
        predictions_path = resolve_project_path(config["paths"]["predictions"])
        metrics_path = resolve_project_path(config["paths"]["metrics"])
        csv_path = resolve_project_path(config["paths"]["per_sample_metrics"])
        training_figure_path = resolve_project_path(config["paths"]["training_curves"])
        rollout_figure_path = resolve_project_path(config["paths"]["rollout_figure"])
        metadata_path = resolve_project_path(config["paths"]["run_metadata"])
        summary_path = resolve_project_path(config["paths"]["summary"])
    else:
        history_path = output_dir / "train_history.json"
        predictions_path = output_dir / "predictions.npz"
        metrics_path = output_dir / "metrics.json"
        csv_path = output_dir / "per_sample_metrics.csv"
        training_figure_path = output_dir / "training_curves.png"
        rollout_figure_path = output_dir / "sample_000_rollout.png"
        metadata_path = output_dir / "run_metadata.json"
        summary_path = output_dir / "summary.md"

    output_dir.mkdir(parents=True, exist_ok=True)
    history_payload = read_json(history_path)
    metrics_payload = read_json(metrics_path)
    if metrics_payload.get("run_type") == "formal" and args.output_dir is not None:
        raise ValueError("Formal result generation must use the configured results directory")
    if history_payload.get("run_type") != metrics_payload.get("run_type"):
        raise ValueError("Training history and inference metrics have different run types")
    records = validate_history(history_payload)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions artifact is missing: {predictions_path}")
    with np.load(predictions_path, allow_pickle=False) as archive:
        required_arrays = {"prediction", "target", "sample_indices", "time_values"}
        missing_arrays = required_arrays.difference(archive.files)
        if missing_arrays:
            raise KeyError(f"Predictions NPZ is missing {sorted(missing_arrays)}")
        prediction = archive["prediction"]
        target = archive["target"]
        sample_indices = archive["sample_indices"]
        time_values = archive["time_values"]

    if prediction.dtype != np.float32 or target.dtype != np.float32:
        raise TypeError("Prediction and target arrays must be float32")
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(f"Invalid prediction/target shapes: {prediction.shape}, {target.shape}")
    if sample_indices.shape != (prediction.shape[0],):
        raise ValueError("sample_indices shape does not match predictions")
    if time_values.shape != (prediction.shape[-1],):
        raise ValueError("time_values shape does not match rollout horizon")
    if not np.array_equal(sample_indices, np.arange(sample_indices[0], sample_indices[0] + len(sample_indices))):
        raise ValueError("sample_indices must be unique, contiguous, and ordered")
    if not np.all(np.diff(time_values.astype(np.float64)) > 0):
        raise ValueError("time_values must be strictly increasing")

    formal = metrics_payload.get("run_type") == "formal"
    if formal:
        expected_shape = (
            int(config["data"]["ntest"]),
            int(config["data"]["resolution"][0]),
            int(config["data"]["resolution"][1]),
            int(config["data"]["horizon"]),
        )
        if prediction.shape != expected_shape:
            raise ValueError(f"Formal prediction shape must be {expected_shape}, got {prediction.shape}")
        expected_indices = np.arange(
            int(config["data"]["test_start"]),
            int(config["data"]["test_start"]) + int(config["data"]["ntest"]),
        )
        if not np.array_equal(sample_indices, expected_indices):
            raise ValueError("Formal sample indices do not match the fixed test split")

    epsilon = float(config["training"]["relative_l2_epsilon"])
    full_metrics, lead_metrics = compute_metrics(prediction, target, epsilon)
    observed_mean = float(metrics_payload["metric"]["full_trajectory_mean"])
    if not np.isclose(full_metrics.mean(), observed_mean, rtol=1e-8, atol=1e-8):
        raise ValueError(
            f"metrics.json full relative L2 mismatch: recomputed={full_metrics.mean()}, "
            f"stored={observed_mean}"
        )
    stored_leads = np.asarray(metrics_payload["metric"]["per_lead_mean"], dtype=np.float64)
    if not np.allclose(lead_metrics.mean(axis=0), stored_leads, rtol=1e-8, atol=1e-8):
        raise ValueError("metrics.json per-lead values do not match predictions")
    verify_csv(csv_path, sample_indices, full_metrics, lead_metrics, time_values)

    best_epoch = int(history_payload["best_epoch"])
    checkpoint_epoch = int(metrics_payload["checkpoint_epoch"])
    if best_epoch != checkpoint_epoch:
        raise ValueError(
            f"History best epoch {best_epoch} does not match checkpoint epoch {checkpoint_epoch}"
        )
    paper_metric = float(config["paper"]["reference_relative_l2"])
    training_figure = make_training_figure(records, paper_metric, best_epoch)
    atomic_save_figure(training_figure, training_figure_path, dpi=300)
    rollout_figure = make_rollout_figure(
        prediction,
        target,
        sample_indices,
        time_values,
        lead_metrics,
        args.sample_index,
    )
    atomic_save_figure(rollout_figure, rollout_figure_path, dpi=300)

    artifact_paths = {
        "train_history": history_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "per_sample_metrics": csv_path,
        "training_curves": training_figure_path,
        "rollout_figure": rollout_figure_path,
    }
    artifact_metadata = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in artifact_paths.items()
    }
    main_metric = float(full_metrics.mean())
    run_metadata = {
        "schema_version": "fno-ns2d-run-metadata-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_type": metrics_payload.get("run_type"),
        "config_path": str(args.config.expanduser().resolve()),
        "data_path": metrics_payload["data_path"],
        "checkpoint_path": metrics_payload["checkpoint_path"],
        "checkpoint_epoch": checkpoint_epoch,
        "test_selected": metrics_payload["test_selected"],
        "split": {
            "train": int(history_payload.get("train_samples", 1000)),
            "validation": 0,
            "test": int(prediction.shape[0]),
        },
        "prediction_shape": list(prediction.shape),
        "normalization": config["data"]["normalization"],
        "metric_formula": metrics_payload["metric"]["formula"],
        "full_trajectory_relative_l2": main_metric,
        "paper_relative_l2": paper_metric,
        "signed_difference": main_metric - paper_metric,
        "absolute_difference": abs(main_metric - paper_metric),
        "parameter_count": metrics_payload["parameter_count"],
        "paper_parameter_count": metrics_payload["paper_parameter_count"],
        "parameter_count_difference": metrics_payload["parameter_count_difference"],
        "runtime": {**metrics_payload["runtime"], "matplotlib": matplotlib.__version__},
        "assumptions": config.get("assumptions", []),
        "conflicts": config.get("conflicts", []),
        "artifacts": artifact_metadata,
        "quality_checks": {
            "all_values_finite": True,
            "metrics_recomputed_from_npz": True,
            "json_metrics_match": True,
            "csv_metrics_match": True,
            "best_epoch_matches_checkpoint": True,
            "figures_nonempty": True,
        },
    }
    atomic_write_json(metadata_path, run_metadata)

    summary = f"""# FNO-2D Navier–Stokes reproduction result

## Summary

- Run type: `{metrics_payload.get('run_type')}`
- Test trajectories: {prediction.shape[0]}
- Forecast shape: `{list(prediction.shape)}`
- Mean full-trajectory relative L2: **{main_metric:.8f}**
- Paper FNO-2D reference (`ν=1e-5`, `T=20`, 1000 train): **{paper_metric:.4f}**
- Absolute difference: **{abs(main_metric-paper_metric):.8f}**
- Checkpoint epoch: {checkpoint_epoch}; selected by train full relative L2 (`test_selected=false`).

## Data and method

The model uses the fixed first 1000 trajectories for training and the final {prediction.shape[0]} trajectories for testing. Ten observed vorticity frames initialize a closed-loop rollout; every predicted frame updates the next input window. No target frame is used after initialization, and no data normalization, padding, augmentation, or PDE-residual loss is applied.

The reported metric is computed per sample as `||prediction-target||₂/(||target||₂+1e-12)` over the full space-time forecast and then averaged. It was recomputed directly from `predictions.npz` and cross-checked against JSON and CSV outputs.

## Reproducibility limitations

The paper does not specify the exact relative-L2 reduction, batch size, random seed, projection hidden width, coordinate-input choice, block ordering, or checkpoint-selection protocol. These choices are recorded explicitly in `config/config.yaml` and `run_metadata.json`. The paper width 32 also cannot be uniquely reconciled with the reported 414,517 parameters from the published connection details; the actual parameter count is reported rather than hidden.

## Artifacts

- `{training_figure_path.name}`: train/test rollout errors and step-loss diagnostics.
- `{rollout_figure_path.name}`: target, prediction, and absolute-error vorticity fields.
- `{predictions_path.name}`: full test prediction and target arrays.
- `{metrics_path.name}` and `{csv_path.name}`: aggregate and per-sample metrics.
- `{metadata_path.name}`: provenance, software, assumptions, hashes, and quality checks.
"""
    atomic_write_text(summary_path, summary)
    print(
        f"result_complete full_relative_l2={main_metric:.8f} "
        f"training_figure={training_figure_path} rollout_figure={rollout_figure_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
