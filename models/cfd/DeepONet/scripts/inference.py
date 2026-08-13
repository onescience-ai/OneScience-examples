#!/usr/bin/env python3
"""Inference for random-test, OOD-curve, and PDE-grid DeepONet cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.DeepONet import build_model  # noqa: E402
from models.dataset import (  # noqa: E402
    OperatorDataset,
    build_split,
    generate_ood_data,
    generate_pde_grid_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--experiment", default="antiderivative")
    parser.add_argument("--variant", default="unstacked_bias")
    parser.add_argument("--mode", choices=("random_test", "ood", "pde_grid"), default="random_test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration {path} must contain a mapping")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_entry(checkpoint_path: Path, experiment: str, variant: str) -> Mapping[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} does not exist. Train the requested entry first; "
            "inference never evaluates random weights."
        )
    bundle = torch_load(checkpoint_path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("entries"), dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is not a DeepONet indexed bundle")
    key = f"{experiment}/{variant}"
    if key not in bundle["entries"]:
        available = sorted(bundle["entries"])
        raise KeyError(f"Checkpoint entry {key!r} is missing; available entries: {available}")
    entry = bundle["entries"][key]
    if not isinstance(entry.get("run_config"), dict):
        raise ValueError(f"Checkpoint entry {key!r} has no resolved run_config")
    return entry


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable")
    return device


def metrics(prediction: np.ndarray, target: np.ndarray, trim_fraction: float) -> Dict[str, float]:
    error = prediction.astype(np.float64).reshape(-1) - target.astype(np.float64).reshape(-1)
    squared = error**2
    output = {
        "test_mse": float(np.mean(squared)),
        "relative_l2": float(
            np.linalg.norm(error)
            / max(np.linalg.norm(target.astype(np.float64).reshape(-1)), np.finfo(np.float64).eps)
        ),
    }
    if trim_fraction > 0.0:
        remove_count = min(len(squared) - 1, int(np.ceil(len(squared) * trim_fraction)))
        kept_count = len(squared) - remove_count
        output["trimmed_test_mse"] = float(np.mean(np.partition(squared, kept_count - 1)[:kept_count]))
    return output


@torch.inference_mode()
def predict_dataset(
    model: torch.nn.Module,
    dataset: OperatorDataset,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, targets = [], []
    model.eval()
    for branch, trunk, target in loader:
        prediction = model(branch.to(device), trunk.to(device))
        predictions.append(prediction.detach().cpu().numpy())
        targets.append(target.numpy())
    prediction_array = np.concatenate(predictions).astype(np.float32)
    target_array = np.concatenate(targets).astype(np.float32)
    if prediction_array.shape != target_array.shape or prediction_array.ndim != 2:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction_array.shape} versus {target_array.shape}"
        )
    if not np.isfinite(prediction_array).all():
        raise FloatingPointError("model prediction contains NaN or infinity")
    return prediction_array, target_array


def _safe_output_dir(
    requested: Path | None,
    results_root: Path,
    experiment: str,
    variant: str,
    mode: str,
) -> Path:
    output = results_root / experiment / variant / mode if requested is None else resolve_path(requested)
    results_root = results_root.resolve()
    output = output.resolve()
    if os.path.commonpath((str(results_root), str(output))) != str(results_root):
        raise ValueError(f"Inference outputs must remain below {results_root}, got {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def final_train_loss(results_root: Path, experiment: str, variant: str) -> float | None:
    history_path = results_root / experiment / variant / "history.csv"
    if not history_path.exists():
        return None
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return float(rows[-1]["train_loss"]) if rows else None


def prepare_data(
    config: Mapping[str, Any], experiment: str, mode: str, seed: int
) -> Tuple[OperatorDataset, Dict[str, np.ndarray]]:
    if mode == "random_test":
        dataset = build_split(config, experiment, "test", PROJECT_ROOT)
        return dataset, {
            "branch_functions": dataset.branch_functions,
            "function_index": dataset.function_index,
            "trunk": dataset.trunk,
        }
    if mode == "ood":
        payload = generate_ood_data(config, experiment)
        dataset = OperatorDataset(payload["branch"], payload["trunk"], payload["target"])
        return dataset, {"branch": payload["branch"], "trunk": payload["trunk"], "labels": payload["labels"]}
    if experiment != "diffusion_reaction":
        raise ValueError("pde_grid mode is only valid for diffusion_reaction")
    payload = generate_pde_grid_case(config, seed)
    dataset = OperatorDataset(payload["branch"], payload["trunk"], payload["target"])
    extras = {key: value for key, value in payload.items() if key != "target"}
    return dataset, extras


def main() -> None:
    args = parse_args()
    file_config = load_yaml(args.config)
    checkpoint_path = resolve_path(args.checkpoint or file_config["paths"]["checkpoint"])
    entry = load_entry(checkpoint_path, args.experiment, args.variant)
    config = entry["run_config"]
    stored_paper_scale = bool(entry.get("paper_scale", config["project"].get("paper_scale", True)))
    if args.smoke_test and stored_paper_scale:
        raise ValueError("--smoke-test was requested, but the selected checkpoint is paper-scale")
    device_name = args.device or str(config["runtime"].get("device", "auto"))
    device = select_device(device_name)
    model = build_model(config, args.experiment, args.variant).to(device)
    model.load_state_dict(entry["model_state"], strict=True)
    seed = int(args.seed if args.seed is not None else config["runtime"]["seed"] + 200_000)
    dataset, extra_arrays = prepare_data(config, args.experiment, args.mode, seed)
    batch_size = int(args.batch_size or config["inference"]["batch_size"])
    prediction, target = predict_dataset(model, dataset, device, batch_size)
    trim_fraction = float(config["experiments"][args.experiment].get("trim_fraction", 0.0))
    result_metrics = metrics(prediction, target, trim_fraction)

    results_root = resolve_path(config["paths"]["results"])
    output_dir = _safe_output_dir(
        args.output_dir, results_root, args.experiment, args.variant, args.mode
    )
    train_loss = final_train_loss(results_root, args.experiment, args.variant)
    if train_loss is not None and args.mode == "random_test":
        result_metrics["generalization_error"] = result_metrics["test_mse"] - train_loss
    metadata = {
        "experiment": args.experiment,
        "variant": args.variant,
        "mode": args.mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(entry["iteration"]),
        "checkpoint_best_metric": float(entry["best_metric"]),
        "seed": seed,
        "paper_scale": stored_paper_scale,
        "sample_count": len(dataset),
    }
    arrays: Dict[str, np.ndarray] = {
        "prediction": prediction,
        "target": target,
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    arrays.update(extra_arrays)
    _atomic_npz(output_dir / "predictions.npz", arrays)
    _atomic_json(output_dir / "metrics.json", {**metadata, **result_metrics})
    metric_text = " ".join(f"{name}={value:.8e}" for name, value in result_metrics.items())
    print(
        f"INFERENCE experiment={args.experiment} variant={args.variant} mode={args.mode} "
        f"paper_scale={stored_paper_scale} {metric_text}",
        flush=True,
    )
    print(
        f"SAVED metrics={output_dir / 'metrics.json'} predictions={output_dir / 'predictions.npz'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
