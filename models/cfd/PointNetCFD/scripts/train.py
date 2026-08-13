#!/usr/bin/env python3
"""Train the paper-faithful PointCFD main experiment."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_string = str(PROJECT_ROOT)
if project_root_string in sys.path:
    sys.path.remove(project_root_string)
sys.path.insert(0, project_root_string)

from models import PointNetCFD, count_trainable_parameters  # noqa: E402
from scripts.common import (  # noqa: E402
    AVAILABLE_SAMPLE_COUNT,
    PAPER_SAMPLE_COUNT,
    append_jsonl,
    checkpoint_metadata_matches,
    choose_device,
    configured_paths,
    evaluate_model,
    load_checkpoint,
    load_config,
    make_loader,
    prepare_datasets,
    resolve_path,
    restore_rng_state,
    rng_state,
    save_checkpoint,
    set_deterministic_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Experiment YAML (default: project config/config.yaml)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override configured batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Override DataLoader workers"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override configured seed")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer, epoch, metrics, and RNG state",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use tiny fixed splits and isolated smoke output paths",
    )
    return parser.parse_args()


def build_effective_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    effective = copy.deepcopy(config)
    training = effective["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs
    elif args.smoke_test:
        training["epochs"] = 1
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    elif args.smoke_test:
        training["batch_size"] = 2
    if args.num_workers is not None:
        training["num_workers"] = args.num_workers
    if args.seed is not None:
        training["seed"] = args.seed
    if int(training["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if int(training["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    if int(training["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    if int(training["validation_interval"]) != 1:
        raise ValueError("The paper validates after every epoch")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("The paper uses Adam")
    if str(training["precision"]).lower() != "float32":
        raise ValueError("This reproduction uses the paper-compatible float32 path")
    return effective


def metric_line(metrics: Dict[str, Any]) -> str:
    rmse = metrics["rmse"]
    relative = metrics["relative_l2"]
    return (
        f"val_mse={metrics['normalized_mse']:.9e} "
        f"rmse_u={rmse['u']:.9e} rmse_v={rmse['v']:.9e} rmse_p={rmse['p']:.9e} "
        f"rel_l2_u={relative['u']['mean']:.9e} "
        f"rel_l2_v={relative['v']['mean']:.9e} "
        f"rel_l2_p={relative['p']['mean']:.9e}"
    )


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    config = build_effective_config(base_config, args)
    training = config["training"]
    seed = int(training["seed"])
    set_deterministic_seed(seed)
    device = choose_device(args.device)

    datasets, target_min, target_max, resolved_paths, split_counts = prepare_datasets(
        config, PROJECT_ROOT, smoke_test=args.smoke_test
    )
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        datasets["train"], batch_size, True, num_workers, seed, pin_memory
    )
    validation_loader = make_loader(
        datasets["validation"], batch_size, False, num_workers, seed, pin_memory
    )

    model = PointNetCFD(
        input_dim=int(config["model"]["input_dim"]),
        output_dim=int(config["model"]["output_dim"]),
    ).to(device=device, dtype=torch.float32)
    parameter_count = count_trainable_parameters(model)
    paper_parameter_count = int(config["model"]["expected_paper_parameters"])
    print(
        f"device={device} trainable_parameters={parameter_count} "
        f"paper_reference_parameters={paper_parameter_count}",
        flush=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.MSELoss(reduction="mean")

    if args.smoke_test:
        checkpoint_path = PROJECT_ROOT / "weight" / "smoke_best_model.pth"
        results_dir = PROJECT_ROOT / "results" / "smoke"
    else:
        checkpoint_path = resolved_paths["checkpoint"]
        results_dir = resolved_paths["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "train_history.jsonl"
    if args.resume is None and history_path.exists():
        history_path.unlink()

    effective_config_path = results_dir / "effective_config.yaml"
    with effective_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    start_epoch = 1
    best_validation_mse = float("inf")
    if args.resume is not None:
        resume_path = resolve_path(PROJECT_ROOT, str(args.resume))
        checkpoint = load_checkpoint(resume_path, device)
        checkpoint_metadata_matches(checkpoint, config, target_min, target_max)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_mse = float(checkpoint["best_validation_mse"])
        restore_rng_state(checkpoint.get("rng_state"))
        if checkpoint.get("train_loader_generator_state") is not None:
            train_loader.generator.set_state(checkpoint["train_loader_generator_state"])
        print(
            f"resumed_from={resume_path} start_epoch={start_epoch} "
            f"best_val_mse={best_validation_mse:.9e}",
            flush=True,
        )

    final_epoch = int(training["epochs"])
    if start_epoch > final_epoch:
        raise ValueError(
            f"Resume checkpoint epoch {start_epoch - 1} already reaches requested epoch {final_epoch}"
        )
    target_names = list(config["data"]["target_names"])
    relative_l2_epsilon = float(config["evaluation"]["relative_l2_epsilon"])
    log_every = int(training["log_every_batches"])
    if log_every <= 0:
        raise ValueError("log_every_batches must be positive")

    run_started = time.time()
    for epoch in range(start_epoch, final_epoch + 1):
        epoch_started = time.time()
        model.train()
        squared_error_sum = 0.0
        element_count = 0
        for batch_number, (coordinates, targets, _) in enumerate(train_loader, start=1):
            coordinates = coordinates.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(coordinates)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_elements = targets.numel()
            squared_error_sum += float(loss.detach().item()) * batch_elements
            element_count += batch_elements
            if batch_number % log_every == 0 or batch_number == len(train_loader):
                print(
                    f"epoch={epoch}/{final_epoch} "
                    f"batch={batch_number}/{len(train_loader)} "
                    f"train_loss={loss.detach().item():.9e}",
                    flush=True,
                )
        train_mse = squared_error_sum / element_count

        validation_metrics, _ = evaluate_model(
            model,
            validation_loader,
            device,
            target_min,
            target_max,
            target_names,
            relative_l2_epsilon,
        )
        validation_mse = float(validation_metrics["normalized_mse"])
        elapsed = time.time() - epoch_started
        print(
            f"epoch={epoch}/{final_epoch} train_mse={train_mse:.9e} "
            f"{metric_line(validation_metrics)} epoch_seconds={elapsed:.3f}",
            flush=True,
        )

        history_record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation": validation_metrics,
            "epoch_seconds": elapsed,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seed": seed,
            "smoke_test": bool(args.smoke_test),
        }
        append_jsonl(history_path, history_record)

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            checkpoint_payload: Dict[str, Any] = {
                "format_version": "pointcfd-checkpoint-v1",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_mse": best_validation_mse,
                "target_min": target_min,
                "target_max": target_max,
                "source_channels": list(config["data"]["source_channels"]),
                "input_names": list(config["data"]["input_names"]),
                "target_names": target_names,
                "input_indices": list(config["data"]["input_indices"]),
                "target_indices": list(config["data"]["target_indices"]),
                "model_config": copy.deepcopy(config["model"]),
                "training_config": copy.deepcopy(training),
                "data_paths": {key: str(value) for key, value in resolved_paths.items()},
                "split_counts": split_counts,
                "seed": seed,
                "available_sample_count": AVAILABLE_SAMPLE_COUNT,
                "paper_sample_count": PAPER_SAMPLE_COUNT,
                "trainable_parameters": parameter_count,
                "rng_state": rng_state(),
                "train_loader_generator_state": train_loader.generator.get_state(),
                "smoke_test": bool(args.smoke_test),
            }
            save_checkpoint(checkpoint_path, checkpoint_payload)
            print(
                f"saved_best_checkpoint={checkpoint_path} "
                f"best_val_mse={best_validation_mse:.9e}",
                flush=True,
            )

    summary = {
        "status": "completed",
        "start_epoch": start_epoch,
        "final_epoch": final_epoch,
        "best_validation_mse": best_validation_mse,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "effective_config": str(effective_config_path),
        "trainable_parameters": parameter_count,
        "paper_reference_parameters": paper_parameter_count,
        "split_counts": split_counts,
        "available_sample_count": AVAILABLE_SAMPLE_COUNT,
        "paper_sample_count": PAPER_SAMPLE_COUNT,
        "dataset_limitation": (
            "The supplied dataset has 2215 cases rather than the paper's 2595; "
            "this is a best-available subset reproduction."
        ),
        "elapsed_seconds": time.time() - run_started,
        "device": str(device),
        "smoke_test": bool(args.smoke_test),
    }
    write_json(results_dir / "training_summary.json", summary)
    print(f"training_complete summary={results_dir / 'training_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
