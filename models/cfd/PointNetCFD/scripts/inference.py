#!/usr/bin/env python3
"""Evaluate a PointCFD checkpoint on the fixed test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_string = str(PROJECT_ROOT)
if project_root_string in sys.path:
    sys.path.remove(project_root_string)
sys.path.insert(0, project_root_string)

from models import PointNetCFD, count_trainable_parameters  # noqa: E402
from scripts.common import (  # noqa: E402
    AVAILABLE_SAMPLE_COUNT,
    PAPER_SAMPLE_COUNT,
    PointCFDDataset,
    choose_device,
    configured_paths,
    evaluate_model,
    load_checkpoint,
    load_config,
    load_data_and_splits,
    make_loader,
    resolve_path,
    selected_indices,
    set_deterministic_seed,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: paths.checkpoint from config)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: paths.results_dir from config)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Evaluate only the first N fixed test cases (smoke testing only)",
    )
    return parser.parse_args()


def validate_checkpoint_contract(checkpoint: Dict[str, Any], config: Dict[str, Any]) -> None:
    expected_metadata = {
        "source_channels": list(config["data"]["source_channels"]),
        "input_names": list(config["data"]["input_names"]),
        "target_names": list(config["data"]["target_names"]),
        "input_indices": list(config["data"]["input_indices"]),
        "target_indices": list(config["data"]["target_indices"]),
    }
    for key, expected in expected_metadata.items():
        if list(checkpoint.get(key, [])) != expected:
            raise ValueError(f"Checkpoint {key} metadata does not match config: {key}")
    target_min = np.asarray(checkpoint.get("target_min"), dtype=np.float32)
    target_max = np.asarray(checkpoint.get("target_max"), dtype=np.float32)
    if target_min.shape != (3,) or target_max.shape != (3,):
        raise ValueError("Checkpoint target normalization must contain three variables")
    if np.any(target_max <= target_min):
        raise ValueError("Checkpoint target normalization spans must be positive")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["training"]["seed"])
    set_deterministic_seed(seed)
    device = choose_device(args.device)
    paths = configured_paths(config, PROJECT_ROOT)
    checkpoint_path = (
        resolve_path(PROJECT_ROOT, str(args.checkpoint))
        if args.checkpoint is not None
        else paths["checkpoint"]
    )
    output_dir = (
        resolve_path(PROJECT_ROOT, str(args.output_dir))
        if args.output_dir is not None
        else paths["results_dir"]
    )

    checkpoint = load_checkpoint(checkpoint_path, device)
    validate_checkpoint_contract(checkpoint, config)
    checkpoint_model_config = checkpoint.get("model_config", config["model"])
    model = PointNetCFD(
        input_dim=int(checkpoint_model_config["input_dim"]),
        output_dim=int(checkpoint_model_config["output_dim"]),
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(
        f"checkpoint={checkpoint_path} epoch={checkpoint.get('epoch')} device={device} "
        f"trainable_parameters={count_trainable_parameters(model)}",
        flush=True,
    )

    data, splits, _ = load_data_and_splits(config, PROJECT_ROOT)
    test_indices = selected_indices(splits["test"], args.max_cases)
    target_min = np.asarray(checkpoint["target_min"], dtype=np.float32)
    target_max = np.asarray(checkpoint["target_max"], dtype=np.float32)
    test_dataset = PointCFDDataset(
        data,
        test_indices,
        config["data"]["input_indices"],
        config["data"]["target_indices"],
        target_min,
        target_max,
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else config["training"]["batch_size"]
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else config["training"]["num_workers"]
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    target_names = list(config["data"]["target_names"])
    metrics, arrays = evaluate_model(
        model,
        test_loader,
        device,
        target_min,
        target_max,
        target_names,
        float(config["evaluation"]["relative_l2_epsilon"]),
    )

    metrics_payload: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": str(device),
        "evaluated_test_cases": len(test_dataset),
        "fixed_test_split_cases": int(splits["test"].size),
        "available_sample_count": AVAILABLE_SAMPLE_COUNT,
        "paper_sample_count": PAPER_SAMPLE_COUNT,
        "metrics": metrics,
        "paper_reference_mean_relative_l2": config["evaluation"][
            "paper_reference_mean_relative_l2"
        ],
        "dataset_limitation": (
            "The supplied dataset has 2215 cases rather than the paper's 2595; "
            "these metrics are a best-available subset reproduction."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "predictions.npz"
    write_json(metrics_path, metrics_payload)
    write_npz(
        predictions_path,
        coordinates=arrays["coordinates"],
        predictions=arrays["predictions"],
        targets=arrays["targets"],
        case_indices=arrays["case_indices"],
        target_names=np.asarray(target_names),
        target_min=target_min,
        target_max=target_max,
    )
    print(
        "normalized_mse={:.9e}".format(metrics["normalized_mse"]), flush=True
    )
    for name in target_names:
        relative = metrics["relative_l2"][name]
        print(
            f"variable={name} rmse={metrics['rmse'][name]:.9e} "
            f"relative_l2_mean={relative['mean']:.9e} "
            f"relative_l2_max={relative['max']:.9e} "
            f"relative_l2_min={relative['min']:.9e}",
            flush=True,
        )
    print(f"metrics={metrics_path}", flush=True)
    print(f"predictions={predictions_path}", flush=True)


if __name__ == "__main__":
    main()
