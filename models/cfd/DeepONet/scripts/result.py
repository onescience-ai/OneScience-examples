#!/usr/bin/env python3
"""Create figures and a summary from existing DeepONet result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--variant", default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration {path} must contain a mapping")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def candidate_runs(results_root: Path, experiment: str | None, variant: str | None) -> Iterable[Tuple[str, str, Path]]:
    experiment_dirs = [results_root / experiment] if experiment else sorted(results_root.iterdir())
    for experiment_dir in experiment_dirs:
        if not experiment_dir.is_dir() or experiment_dir.name in {"cache", "figures"}:
            continue
        variant_dirs = [experiment_dir / variant] if variant else sorted(experiment_dir.iterdir())
        for variant_dir in variant_dirs:
            if variant_dir.is_dir():
                yield experiment_dir.name, variant_dir.name, variant_dir


def read_history(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"History {path} is empty")
    columns: Dict[str, np.ndarray] = {}
    for key in rows[0]:
        if key == "paper_scale":
            continue
        try:
            columns[key] = np.asarray([float(row[key]) for row in rows if row.get(key, "") != ""])
        except ValueError:
            continue
    return columns


def plot_history(history_path: Path, output_path: Path, title: str) -> None:
    columns = read_history(history_path)
    iterations = columns["iteration"]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(iterations, columns["train_loss"], label="train MSE", linewidth=1.8)
    axis.plot(iterations, columns["test_mse"], label="test MSE", linewidth=1.8)
    if "trimmed_test_mse" in columns:
        axis.plot(iterations, columns["trimmed_test_mse"], label="trimmed test MSE", linewidth=1.5)
    axis.set_yscale("log")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Mean squared error")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _metadata(payload: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    return json.loads(str(payload["metadata"].item())) if "metadata" in payload else {}


def plot_ood(payload: Mapping[str, np.ndarray], output_path: Path, title: str) -> None:
    labels = payload["labels"].astype(str)
    trunk = payload["trunk"][:, 0]
    target = payload["target"][:, 0]
    prediction = payload["prediction"][:, 0]
    unique_labels = list(dict.fromkeys(labels.tolist()))
    fig, axes = plt.subplots(len(unique_labels), 1, figsize=(7.2, 3.0 * len(unique_labels)), squeeze=False)
    for axis, label in zip(axes[:, 0], unique_labels):
        selected = labels == label
        order = np.argsort(trunk[selected])
        axis.plot(trunk[selected][order], target[selected][order], label="reference", linewidth=2.0)
        axis.plot(trunk[selected][order], prediction[selected][order], "--", label="DeepONet", linewidth=1.8)
        axis.set_title(label)
        axis.set_xlabel("Query coordinate")
        axis.set_ylabel("Solution")
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pde(payload: Mapping[str, np.ndarray], output_path: Path, title: str) -> None:
    shape = tuple(int(value) for value in payload["grid_shape"])
    target = payload["target"].reshape(shape)
    prediction = payload["prediction"].reshape(shape)
    absolute_error = np.abs(prediction - target)
    x_grid = payload["x"]
    t_grid = payload["t"]
    extent = (float(x_grid.min()), float(x_grid.max()), float(t_grid.min()), float(t_grid.max()))
    shared_min = float(min(target.min(), prediction.min()))
    shared_max = float(max(target.max(), prediction.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    first = axes[0].imshow(target, origin="lower", aspect="auto", extent=extent, vmin=shared_min, vmax=shared_max)
    axes[0].set_title("Reference")
    axes[1].imshow(prediction, origin="lower", aspect="auto", extent=extent, vmin=shared_min, vmax=shared_max)
    axes[1].set_title("Prediction")
    error_image = axes[2].imshow(absolute_error, origin="lower", aspect="auto", extent=extent)
    axes[2].set_title("Absolute error")
    for axis in axes:
        axis.set_xlabel("x")
        axis.set_ylabel("t")
    fig.colorbar(first, ax=axes[:2], shrink=0.82, label="s(x,t)")
    fig.colorbar(error_image, ax=axes[2], shrink=0.82, label="absolute error")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_generic(payload: Mapping[str, np.ndarray], output_path: Path, title: str) -> None:
    prediction = payload["prediction"].reshape(-1)
    target = payload["target"].reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    limit_min = float(min(prediction.min(), target.min()))
    limit_max = float(max(prediction.max(), target.max()))
    axes[0].scatter(target, prediction, s=7, alpha=0.45)
    axes[0].plot((limit_min, limit_max), (limit_min, limit_max), "k--", linewidth=1)
    axes[0].set_xlabel("Reference")
    axes[0].set_ylabel("Prediction")
    axes[0].set_title("Prediction parity")
    axes[1].hist(np.abs(prediction - target), bins=40)
    axes[1].set_xlabel("Absolute error")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error distribution")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, records: List[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": records}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results_root = resolve_path(config["paths"]["results"])
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory {results_root} does not exist")
    figures_root = results_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    records: List[Mapping[str, Any]] = []
    artifact_count = 0
    for experiment, variant, run_dir in candidate_runs(results_root, args.experiment, args.variant):
        title = f"{experiment} / {variant}"
        history_path = run_dir / "history.csv"
        if history_path.exists():
            output = figures_root / f"{experiment}_{variant}_loss.png"
            plot_history(history_path, output, title)
            print(f"FIGURE {output}", flush=True)
            artifact_count += 1
        mode_dirs = sorted(
            path
            for path in run_dir.iterdir()
            if path.is_dir() and ((path / "predictions.npz").exists() or (path / "metrics.json").exists())
        )
        # Backward-compatible fallback for results created before mode directories were introduced.
        artifact_dirs = mode_dirs or [run_dir]
        for artifact_dir in artifact_dirs:
            prediction_path = artifact_dir / "predictions.npz"
            metadata: Dict[str, Any] = {}
            if prediction_path.exists():
                with np.load(prediction_path, allow_pickle=False) as payload_file:
                    payload = {key: payload_file[key] for key in payload_file.files}
                metadata = _metadata(payload)
                suffix = str(metadata.get("mode", artifact_dir.name))
                output = figures_root / f"{experiment}_{variant}_{suffix}.png"
                mode_title = f"{title} / {suffix}"
                if "labels" in payload:
                    plot_ood(payload, output, mode_title)
                elif "grid_shape" in payload:
                    plot_pde(payload, output, mode_title)
                else:
                    plot_generic(payload, output, mode_title)
                print(f"FIGURE {output}", flush=True)
                artifact_count += 1
            metrics_path = artifact_dir / "metrics.json"
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                records.append(record)
                metric_names = ("test_mse", "trimmed_test_mse", "relative_l2", "generalization_error")
                metric_text = " ".join(
                    f"{name}={float(record[name]):.8e}" for name in metric_names if name in record
                )
                print(
                    f"METRICS experiment={experiment} variant={variant} "
                    f"mode={record.get('mode', metadata.get('mode', artifact_dir.name))} {metric_text}",
                    flush=True,
                )
    if artifact_count == 0 and not records:
        raise FileNotFoundError("No history.csv, predictions.npz, or metrics.json matched the request")
    summary_path = figures_root / "summary.json"
    write_summary(summary_path, records)
    print(f"SUMMARY {summary_path} runs={len(records)} figures={artifact_count}", flush=True)


if __name__ == "__main__":
    main()
