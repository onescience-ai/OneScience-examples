#!/usr/bin/env python3
"""Run strict closed-loop inference for the trained FNO-2D checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import build_model_from_config  # noqa: E402
from train import (  # noqa: E402
    atomic_write_json,
    build_datasets,
    build_loader,
    data_file_from_config,
    environment_metadata,
    load_config,
    resolve_project_path,
    seed_everything,
    select_device,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FNO-2D checkpoint on the fixed test split."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None, help="Smoke only.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Smoke only.")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint root must be a mapping")
    required = {
        "model_state_dict",
        "config",
        "epoch",
        "parameter_count",
        "monitor",
        "test_selected",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"Checkpoint is missing required keys: {missing}")
    if checkpoint["test_selected"] is not False:
        raise ValueError("This reproduction forbids a test-selected checkpoint")
    if checkpoint["monitor"] != "train_full_relative_l2":
        raise ValueError(f"Unexpected checkpoint monitor: {checkpoint['monitor']}")
    return checkpoint


def nested_value(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        value = value[key]
    return value


def validate_checkpoint_config(
    current: dict[str, Any], checkpoint_config: dict[str, Any]
) -> None:
    keys = (
        "data.key",
        "data.layout",
        "data.dtype",
        "data.expected_shape",
        "data.resolution",
        "data.ntrain",
        "data.ntest",
        "data.test_start",
        "data.history",
        "data.horizon",
        "data.normalization",
        "model.input_channels",
        "model.output_channels",
        "model.use_grid",
        "model.grid_include_endpoint",
        "model.width",
        "model.modes1",
        "model.modes2",
        "model.num_layers",
        "model.projection_width",
        "model.fft_norm",
        "training.dtype",
        "training.relative_l2_epsilon",
    )
    differences = []
    for key in keys:
        current_value = nested_value(current, key)
        checkpoint_value = nested_value(checkpoint_config, key)
        if current_value != checkpoint_value:
            differences.append(f"{key}: current={current_value!r}, checkpoint={checkpoint_value!r}")
    if differences:
        raise ValueError("Checkpoint/config mismatch:\n" + "\n".join(differences))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compute_metrics(
    prediction: np.ndarray, target: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {prediction.shape} vs {target.shape}")
    if prediction.ndim != 4:
        raise ValueError(f"Expected [N,H,W,T], received {prediction.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("Prediction or target contains NaN/Inf")

    difference = prediction.astype(np.float64) - target.astype(np.float64)
    target64 = target.astype(np.float64)
    full_numerator = np.linalg.norm(difference.reshape(prediction.shape[0], -1), axis=1)
    full_denominator = np.linalg.norm(target64.reshape(target.shape[0], -1), axis=1)
    full = full_numerator / (full_denominator + epsilon)

    difference_by_time = np.moveaxis(difference, -1, 1).reshape(
        prediction.shape[0], prediction.shape[-1], -1
    )
    target_by_time = np.moveaxis(target64, -1, 1).reshape(
        target.shape[0], target.shape[-1], -1
    )
    per_lead = np.linalg.norm(difference_by_time, axis=2) / (
        np.linalg.norm(target_by_time, axis=2) + epsilon
    )
    return full, per_lead


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path,
    sample_indices: np.ndarray,
    full_metrics: np.ndarray,
    lead_metrics: np.ndarray,
    time_values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    header = ["sample_index", "relative_l2_full"] + [
        f"relative_l2_t{int(value)}" for value in time_values
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row, sample_index in enumerate(sample_indices):
            writer.writerow(
                [int(sample_index), f"{full_metrics[row]:.17g}"]
                + [f"{value:.17g}" for value in lead_metrics[row]]
            )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    inference = config["inference"]
    training = config["training"]
    seed = int(inference.get("seed", training["seed"]))
    seed_everything(seed, bool(training.get("deterministic", True)))
    device = select_device(args.device)
    checkpoint_path = (
        resolve_project_path(config["paths"]["checkpoint"])
        if args.checkpoint is None
        else args.checkpoint.expanduser().resolve()
    )
    output_dir = (
        resolve_project_path(config["paths"]["results_dir"])
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, device)
    validate_checkpoint_config(config, checkpoint["config"])

    checkpoint_run_type = str(checkpoint.get("run_type", "formal"))
    if checkpoint_run_type == "formal" and (
        args.max_test_samples is not None
        or args.rollout_steps is not None
        or args.output_dir is not None
        or args.checkpoint is not None
    ):
        raise ValueError(
            "Formal inference uses the exact configured checkpoint, test split, horizon, "
            "and results path; overrides are only allowed for smoke checkpoints"
        )
    horizon = int(config["data"]["horizon"])
    rollout_steps = horizon if args.rollout_steps is None else int(args.rollout_steps)
    if not 1 <= rollout_steps <= horizon:
        raise ValueError(f"rollout_steps must be in [1,{horizon}]")

    _, test_dataset = build_datasets(
        config,
        max_train_samples=1,
        max_test_samples=args.max_test_samples,
        rollout_steps=rollout_steps,
    )
    batch_size = int(
        inference["batch_size"] if args.batch_size is None else args.batch_size
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", True)) and device.type == "cuda",
        seed=seed,
    )

    # Preserve complex64 spectral parameters while moving the model to device.
    model = build_model_from_config(config).to(device=device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict state load failed: {incompatible}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(checkpoint["parameter_count"]):
        raise ValueError(
            f"Parameter count mismatch: model={parameter_count}, "
            f"checkpoint={checkpoint['parameter_count']}"
        )
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    synchronize(device)
    started = time.perf_counter()
    processed = 0
    with torch.inference_mode():
        for batch_number, (history, target) in enumerate(test_loader, start=1):
            history = history.to(device=device, dtype=torch.float32, non_blocking=True)
            target_device = target.to(device=device, dtype=torch.float32, non_blocking=True)
            window = history
            batch_prediction: list[torch.Tensor] = []
            for step in range(rollout_steps):
                prediction_step = model(window)
                if not torch.isfinite(prediction_step).all():
                    raise FloatingPointError(
                        f"Non-finite prediction at batch {batch_number}, step {step + 1}"
                    )
                batch_prediction.append(prediction_step)
                window = torch.cat((window[..., 1:], prediction_step), dim=-1)
            prediction = torch.cat(batch_prediction, dim=-1)
            predictions.append(prediction.cpu().numpy().astype(np.float32, copy=False))
            targets.append(target_device.cpu().numpy().astype(np.float32, copy=False))
            processed += int(history.shape[0])
            print(
                f"inference_batch={batch_number:03d}/{len(test_loader):03d} "
                f"processed={processed}/{len(test_dataset)}",
                flush=True,
            )
    synchronize(device)
    duration = time.perf_counter() - started

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    expected_shape = (
        len(test_dataset),
        int(config["data"]["resolution"][0]),
        int(config["data"]["resolution"][1]),
        rollout_steps,
    )
    if prediction_array.shape != expected_shape or target_array.shape != expected_shape:
        raise ValueError(
            f"Unexpected inference arrays: prediction={prediction_array.shape}, "
            f"target={target_array.shape}, expected={expected_shape}"
        )
    epsilon = float(training["relative_l2_epsilon"])
    full_metrics, lead_metrics = compute_metrics(prediction_array, target_array, epsilon)
    test_start = int(config["data"]["test_start"])
    sample_indices = np.arange(
        test_start, test_start + len(test_dataset), dtype=np.int64
    )
    configured_times = np.asarray(config["data"]["future_times"], dtype=np.float32)
    time_values = configured_times[:rollout_steps]

    if args.output_dir is None:
        predictions_path = resolve_project_path(config["paths"]["predictions"])
        metrics_path = resolve_project_path(config["paths"]["metrics"])
        csv_path = resolve_project_path(config["paths"]["per_sample_metrics"])
    else:
        predictions_path = output_dir / "predictions.npz"
        metrics_path = output_dir / "metrics.json"
        csv_path = output_dir / "per_sample_metrics.csv"

    atomic_save_npz(
        predictions_path,
        prediction=prediction_array,
        target=target_array,
        sample_indices=sample_indices,
        time_values=time_values,
    )
    atomic_write_csv(csv_path, sample_indices, full_metrics, lead_metrics, time_values)

    paper_metric = float(config["paper"]["reference_relative_l2"])
    mean_full = float(full_metrics.mean())
    metrics_payload: dict[str, Any] = {
        "schema_version": "fno-ns2d-metrics-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_type": checkpoint_run_type,
        "config_path": str(args.config.expanduser().resolve()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_monitor": checkpoint["monitor"],
        "test_selected": bool(checkpoint["test_selected"]),
        "data_path": str(data_file_from_config(config)),
        "prediction_path": str(predictions_path),
        "per_sample_metrics_path": str(csv_path),
        "sample_count": int(len(test_dataset)),
        "prediction_shape": list(prediction_array.shape),
        "sample_indices": {"first": int(sample_indices[0]), "last": int(sample_indices[-1])},
        "time_values": [float(value) for value in time_values],
        "metric": {
            "name": "samplewise_relative_l2",
            "formula": "||prediction-target||_2/(||target||_2+epsilon), then arithmetic mean over samples",
            "epsilon": epsilon,
            "full_trajectory_mean": mean_full,
            "full_trajectory_std": float(full_metrics.std(ddof=0)),
            "mean_step_relative_l2": float(lead_metrics.mean()),
            "per_lead_mean": [float(value) for value in lead_metrics.mean(axis=0)],
            "per_lead_std": [float(value) for value in lead_metrics.std(axis=0, ddof=0)],
        },
        "paper_comparison": {
            "paper_relative_l2": paper_metric,
            "signed_difference": mean_full - paper_metric,
            "absolute_difference": abs(mean_full - paper_metric),
        },
        "parameter_count": parameter_count,
        "paper_parameter_count": int(config["paper"]["reference_parameter_count"]),
        "parameter_count_difference": parameter_count
        - int(config["paper"]["reference_parameter_count"]),
        "runtime": {
            **environment_metadata(device),
            "duration_seconds": duration,
            "batch_size": batch_size,
        },
        "assumptions": config.get("assumptions", []),
    }
    atomic_write_json(metrics_path, metrics_payload)
    print(
        f"inference_complete samples={len(test_dataset)} duration={duration:.3f}s "
        f"full_relative_l2={mean_full:.8f} paper={paper_metric:.8f} "
        f"absolute_difference={abs(mean_full-paper_metric):.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
