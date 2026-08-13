#!/usr/bin/env python3
"""Evaluate a trained CNO on the paper's ID and OOD Navier--Stokes sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.FNO import build_model
from scripts.common import (
    MinMaxNormalizer,
    NavierStokesH5Dataset,
    atomic_json_dump,
    atomic_npz_save,
    data_file,
    load_config,
    numeric_sample_ids,
    project_path,
    relative_l1_per_sample,
    select_device,
    set_reproducibility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "config.yaml")
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("id", "ood"),
        default=("id", "ood"),
    )
    return parser.parse_args()


def load_trained_model(
    config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, MinMaxNormalizer, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {
        "model_state_dict",
        "normalization",
        "epoch",
        "best_val_relative_l1",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"checkpoint is missing required keys: {missing}")
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config and checkpoint_config.get("model") != config["model"]:
        raise ValueError("checkpoint model configuration differs from config.yaml")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    normalizer = MinMaxNormalizer.from_state(checkpoint["normalization"])
    return model, normalizer, checkpoint


@torch.inference_mode()
def evaluate_split(
    split_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: MinMaxNormalizer,
    device: torch.device,
    epsilon: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_ids: list[np.ndarray] = []
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_ratios: list[np.ndarray] = []
    completed = 0
    total = len(loader.dataset)

    for batch_index, (inputs, targets, sample_ids) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs)
        physical_inputs = normalizer.denormalize_input(inputs)
        physical_targets = normalizer.denormalize_output(targets)
        physical_predictions = normalizer.denormalize_output(predictions)
        ratios = relative_l1_per_sample(
            physical_predictions, physical_targets, epsilon
        )

        all_ids.append(np.asarray(sample_ids, dtype=np.int64))
        all_inputs.append(physical_inputs.cpu().numpy().astype(np.float32))
        all_targets.append(physical_targets.cpu().numpy().astype(np.float32))
        all_predictions.append(physical_predictions.cpu().numpy().astype(np.float32))
        all_ratios.append(ratios.cpu().numpy().astype(np.float64))
        completed += inputs.shape[0]
        running = np.concatenate(all_ratios) * 100.0
        print(
            f"inference split={split_name} batch={batch_index}/{len(loader)} "
            f"samples={completed}/{total} running_rel_l1_median={np.median(running):.6f}%",
            flush=True,
        )

    arrays = {
        "sample_ids": np.concatenate(all_ids),
        "inputs": np.concatenate(all_inputs),
        "targets": np.concatenate(all_targets),
        "predictions": np.concatenate(all_predictions),
        "relative_l1": np.concatenate(all_ratios),
    }
    percentages = arrays["relative_l1"] * 100.0
    metrics = {
        "sample_count": int(percentages.size),
        "resolution": [int(arrays["inputs"].shape[-2]), int(arrays["inputs"].shape[-1])],
        "relative_l1_median_percent": float(np.median(percentages)),
        "relative_l1_mean_percent": float(np.mean(percentages)),
        "relative_l1_std_percent": float(np.std(percentages)),
        "relative_l1_min_percent": float(np.min(percentages)),
        "relative_l1_max_percent": float(np.max(percentages)),
    }
    return arrays, metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    inference_config = config["inference"]
    device = select_device(args.device or str(inference_config["device"]))
    batch_size = int(
        args.batch_size if args.batch_size is not None else inference_config["batch_size"]
    )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    seed = int(config["experiment"]["seed"])
    set_reproducibility(seed, bool(config["experiment"].get("deterministic", True)))

    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else project_path(config["paths"]["checkpoint"])
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_path(config["paths"]["results_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model, normalizer, checkpoint = load_trained_model(config, checkpoint_path, device)

    split_specs = {
        "id": ("id_test_file", "test_id"),
        "ood": ("ood_test_file", "test_ood"),
    }
    all_metrics: dict[str, Any] = {
        "schema_version": "cno-navier-stokes-metrics-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_relative_l1_percent": float(
            checkpoint["best_val_relative_l1"]
        ),
        "normalization": normalizer.state_dict(),
        "paper_reference": config.get("paper_reference", {}),
        "splits": {},
    }
    print(
        f"inference device={device} checkpoint_epoch={checkpoint['epoch']} "
        f"best_val={float(checkpoint['best_val_relative_l1']):.6f}%",
        flush=True,
    )

    for split_name in args.splits:
        filename_key, split_key = split_specs[split_name]
        dataset = NavierStokesH5Dataset(
            data_file(config, filename_key),
            numeric_sample_ids(config["data"][split_key]),
            normalizer,
            str(config["data"]["input_key"]),
            str(config["data"]["output_key"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(inference_config["num_workers"]),
            pin_memory=device.type == "cuda",
            persistent_workers=int(inference_config["num_workers"]) > 0,
        )
        arrays, metrics = evaluate_split(
            split_name,
            model,
            loader,
            normalizer,
            device,
            float(inference_config["metric_epsilon"]),
        )
        artifact_path = output_dir / f"{split_name}_predictions.npz"
        atomic_npz_save(artifact_path, **arrays)
        metrics["predictions_file"] = str(artifact_path)
        all_metrics["splits"][split_name] = metrics
        print(
            f"evaluation split={split_name} n={metrics['sample_count']} "
            f"rel_l1_median={metrics['relative_l1_median_percent']:.6f}% "
            f"mean={metrics['relative_l1_mean_percent']:.6f}% "
            f"std={metrics['relative_l1_std_percent']:.6f}% "
            f"saved={artifact_path}",
            flush=True,
        )

    metrics_path = output_dir / "metrics.json"
    atomic_json_dump(all_metrics, metrics_path)
    print(f"metrics saved path={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
