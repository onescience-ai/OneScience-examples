"""Autoregressive MP-PDE E3 rollout and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataset import E3Dataset  # noqa: E402
from scripts.train import build_model, choose_device, load_config, project_path, rollout_batch, set_seed  # noqa: E402


CRITICAL_CONFIG_KEYS = (
    ("model", "time_window"),
    ("model", "hidden_dim"),
    ("model", "message_passing_layers"),
    ("model", "neighbor_offsets"),
    ("model", "aggregation"),
    ("model", "decoder", "kernel_sizes"),
    ("model", "decoder", "strides"),
    ("model", "scaling", "coordinates"),
    ("model", "scaling", "parameters"),
    ("data", "equation", "alpha_range"),
    ("data", "equation", "beta_range"),
    ("data", "equation", "gamma_range"),
)


def _nested(mapping: Mapping[str, Any], key: Iterable[str]) -> Any:
    value: Any = mapping
    for component in key:
        value = value[component]
    return value


def verify_checkpoint_config(runtime: Mapping[str, Any], stored: Mapping[str, Any]) -> None:
    mismatches = []
    for key in CRITICAL_CONFIG_KEYS:
        runtime_value, stored_value = _nested(runtime, key), _nested(stored, key)
        if runtime_value != stored_value:
            mismatches.append(f"{'.'.join(key)}: runtime={runtime_value!r}, checkpoint={stored_value!r}")
    if mismatches:
        raise ValueError("Checkpoint/config architecture mismatch:\n  " + "\n  ".join(mismatches))


def compute_metrics(prediction: np.ndarray, target: np.ndarray, forecast_start: int) -> Dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(f"prediction/target must share [S,T,N] shape, found {prediction.shape}/{target.shape}")
    error = prediction[:, forecast_start:] - target[:, forecast_start:]
    if error.size == 0 or not np.all(np.isfinite(error)):
        raise FloatingPointError("Forecast error is empty or non-finite")
    squared = error.astype(np.float64) ** 2
    per_time_mse = np.mean(squared, axis=(0, 2))
    numerator = np.linalg.norm(error.reshape(error.shape[0], -1), axis=1)
    denominator = np.linalg.norm(target[:, forecast_start:].reshape(error.shape[0], -1), axis=1) + 1.0e-12
    return {
        "accumulated_mse": float(np.sum(per_time_mse)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": float(np.mean(numerator / denominator)),
        "per_time_mse": per_time_mse,
    }


def atomic_npz(destination: Path, **arrays: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.stem + ".", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MP-PDE E3 autoregressive inference")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", type=str)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    inference_cfg = config["inference"]
    checkpoint_path = args.checkpoint or project_path(config["paths"]["checkpoint"])
    data_path = args.data or project_path(config["paths"]["data"])
    output_dir = args.output_dir or project_path(config["paths"]["results"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Real trained checkpoint not found: {checkpoint_path}. Run scripts/train.py first.")
    if not data_path.is_file():
        raise FileNotFoundError(f"E3 HDF5 data not found: {data_path}. Generate data before inference.")
    seed = int(args.seed if args.seed is not None else config["training"]["seed"])
    set_seed(seed, bool(config["training"]["deterministic"]))
    device = choose_device(args.device or str(inference_cfg["device"]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint or "resolved_config" not in checkpoint:
        raise KeyError("Checkpoint must contain model_state and resolved_config")
    verify_checkpoint_config(config, checkpoint["resolved_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    split = args.split or str(inference_cfg["split"])
    dataset = E3Dataset(
        data_path, split, expected_nt=int(config["data"]["num_time_points"]), expected_nx=int(config["data"]["resolution"])
    )
    batch_size = int(args.batch_size or inference_cfg["batch_size"])
    configured_max = inference_cfg.get("max_samples")
    max_samples = args.max_samples if args.max_samples is not None else configured_max
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    window = int(config["model"]["time_window"])
    predictions, targets, parameters, sample_indices = [], [], [], []
    first_x, first_t = None, None
    processed = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if max_samples is not None:
                remaining = int(max_samples) - processed
                if remaining <= 0:
                    break
                if batch["u"].shape[0] > remaining:
                    batch = {key: value[:remaining] for key, value in batch.items()}
            prediction = rollout_batch(model, batch, device, window).cpu().numpy()
            predictions.append(prediction)
            targets.append(batch["u"].numpy())
            parameters.append(batch["params"].numpy())
            sample_indices.append(batch["index"].numpy())
            if first_x is None:
                first_x, first_t = batch["x"][0].numpy(), batch["t"][0].numpy()
            processed += prediction.shape[0]
            print(f"inference batch={batch_index} processed={processed}", flush=True)
    if not predictions or first_x is None or first_t is None:
        raise RuntimeError("Inference produced no samples")
    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    parameter_array = np.concatenate(parameters)
    index_array = np.concatenate(sample_indices)
    forecast_start = int(inference_cfg["forecast_start_index"])
    if forecast_start != window:
        raise ValueError("E3 inference forecast_start_index must equal model.time_window")
    if not np.array_equal(prediction_array[:, :forecast_start], target_array[:, :forecast_start]):
        raise AssertionError("Rollout seed differs from the first K ground-truth states")
    metrics = compute_metrics(prediction_array, target_array, forecast_start)
    per_time_mse = metrics.pop("per_time_mse")
    reference_values = config["paper_reference"]["values"]
    nx = int(prediction_array.shape[2])
    paper_reference = reference_values.get(nx, reference_values.get(str(nx)))
    metrics_payload: Dict[str, Any] = {
        **metrics,
        "samples": int(prediction_array.shape[0]),
        "split": split,
        "nx": nx,
        "nt": int(prediction_array.shape[1]),
        "forecast_start_index": forecast_start,
        "forecast_end_index_inclusive": int(prediction_array.shape[1] - 1),
        "metric_scope": "forecast_only; accumulated_mse=sum_time(mean_sample_space(error^2))",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_validation_accumulated_mse": float(checkpoint.get("best_validation_accumulated_mse", float("nan"))),
        "ambiguity_policy": config["experiment"]["ambiguity_policy"],
        "paper_reference": {"value": paper_reference, "provenance_only_not_assertion": True},
    }
    predictions_path = output_dir / Path(config["paths"]["predictions"]).name
    metrics_path = output_dir / Path(config["paths"]["metrics"]).name
    atomic_npz(
        predictions_path,
        prediction=prediction_array,
        target=target_array,
        x=first_x,
        t=first_t,
        params=parameter_array,
        sample_indices=index_array,
        forecast_start_index=np.asarray(forecast_start, dtype=np.int64),
        per_time_mse=per_time_mse,
    )
    atomic_json(metrics_path, metrics_payload)
    print(
        f"metrics samples={metrics_payload['samples']} accumulated_mse={metrics['accumulated_mse']:.8e} "
        f"rmse={metrics['rmse']:.8e} mae={metrics['mae']:.8e} relative_l2={metrics['relative_l2']:.8e}",
        flush=True,
    )
    print(f"Saved predictions: {predictions_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
