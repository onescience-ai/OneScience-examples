"""Create MP-PDE E3 visualizations from real inference/training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real MP-PDE E3 rollout results")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-index", type=int)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    predictions_path = args.predictions or project_path(config["paths"]["predictions"])
    metrics_path = args.metrics or project_path(config["paths"]["metrics"])
    history_path = args.history or project_path(config["paths"]["train_history"])
    output_dir = args.output_dir or project_path(config["paths"]["results"])
    if not predictions_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Real inference artifacts are required: {predictions_path} and {metrics_path}. "
            "Run scripts/inference.py after training; placeholder data will not be generated."
        )
    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {"prediction", "target", "x", "t", "params", "sample_indices", "forecast_start_index", "per_time_mse"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"predictions.npz is missing fields: {sorted(missing)}")
        prediction, target = archive["prediction"], archive["target"]
        x, t, per_time_mse = archive["x"], archive["t"], archive["per_time_mse"]
        forecast_start = int(archive["forecast_start_index"])
    with metrics_path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(f"Expected matching [S,T,N] arrays, found {prediction.shape}/{target.shape}")
    if prediction.shape[1:] != (t.size, x.size) or not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("Prediction axes do not match x/t or contain non-finite values")
    recomputed = np.mean((prediction[:, forecast_start:] - target[:, forecast_start:]) ** 2, axis=(0, 2))
    if per_time_mse.shape != recomputed.shape or not np.allclose(per_time_mse, recomputed, rtol=2e-5, atol=1e-8):
        raise ValueError("Stored per_time_mse is inconsistent with prediction and target")
    if not np.isclose(float(metrics["accumulated_mse"]), float(np.sum(recomputed)), rtol=2e-5, atol=1e-8):
        raise ValueError("metrics.json accumulated_mse is inconsistent with predictions.npz")

    output_dir.mkdir(parents=True, exist_ok=True)
    sample = int(args.sample_index if args.sample_index is not None else config["visualization"]["sample_index"])
    if sample < 0 or sample >= prediction.shape[0]:
        raise IndexError(f"sample_index={sample} outside [0,{prediction.shape[0]})")
    dpi = int(config["visualization"]["dpi"])
    time_indices = [int(index) for index in config["visualization"]["time_indices"]]
    if any(index < 0 or index >= t.size for index in time_indices):
        raise IndexError(f"Configured time_indices exceed nt={t.size}")

    figure, axes = plt.subplots(len(time_indices), 1, figsize=(9, 2.4 * len(time_indices)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, time_index in zip(axes, time_indices):
        axis.plot(x, target[sample, time_index], color="black", linewidth=1.5, label="target")
        axis.plot(x, prediction[sample, time_index], color="tab:blue", linewidth=1.2, linestyle="--", label="MP-PDE")
        axis.set_ylabel("u")
        axis.set_title(f"t={t[time_index]:.4f}, index={time_index}")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("x")
    figure.tight_layout()
    rollout_path = output_dir / "e3_rollout.png"
    figure.savefig(rollout_path, dpi=dpi)
    plt.close(figure)

    absolute_error = np.abs(prediction[sample] - target[sample])
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2.2, 1.0]})
    image = axes[0].imshow(
        absolute_error.T, origin="lower", aspect="auto", extent=(float(t[0]), float(t[-1]), float(x[0]), float(x[-1])), cmap="magma"
    )
    axes[0].axvline(float(t[forecast_start]), color="white", linestyle="--", linewidth=1.0, label="forecast start")
    axes[0].set_ylabel("x")
    axes[0].set_title("Absolute rollout error")
    axes[0].legend(loc="upper right")
    figure.colorbar(image, ax=axes[0], label="|prediction-target|")
    axes[1].plot(t[forecast_start:], per_time_mse, color="tab:red")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("MSE")
    axes[1].set_title(f"Per-time MSE; accumulated={metrics['accumulated_mse']:.6g}")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    error_path = output_dir / "e3_error.png"
    figure.savefig(error_path, dpi=dpi)
    plt.close(figure)

    created = [rollout_path, error_path]
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as stream:
            history = json.load(stream)
        if not isinstance(history, list) or not history:
            raise ValueError(f"Training history is empty or malformed: {history_path}")
        epochs = [int(item["epoch"]) + 1 for item in history]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [item["train_rmse"] for item in history], label="train bundle RMSE")
        axes[0].plot(epochs, [item["validation_bundle_rmse"] for item in history], label="validation bundle RMSE")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("RMSE")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        axes[1].plot(epochs, [item["validation_accumulated_mse"] for item in history], color="tab:purple")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("validation accumulated MSE")
        axes[1].grid(alpha=0.2)
        figure.tight_layout()
        training_path = output_dir / "training_curve.png"
        figure.savefig(training_path, dpi=dpi)
        plt.close(figure)
        created.append(training_path)
    else:
        print(f"Training history not found; skipped training curve: {history_path}", flush=True)
    for path in created:
        print(f"Saved figure: {path}", flush=True)


if __name__ == "__main__":
    main()
