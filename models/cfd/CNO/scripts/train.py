#!/usr/bin/env python3
"""Train the paper-configured CNO on the 2-D Navier--Stokes benchmark."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.FNO import build_model, count_trainable_parameters
from scripts.common import (
    MinMaxNormalizer,
    NavierStokesH5Dataset,
    atomic_json_dump,
    atomic_torch_save,
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
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="experiment YAML configuration",
    )
    parser.add_argument("--device", default=None, help="override training.device")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument(
        "--resume",
        default=None,
        help="resume a complete training state; not a weight-only initialization",
    )
    return parser.parse_args()


def _make_loader(
    dataset: NavierStokesH5Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: MinMaxNormalizer,
    device: torch.device,
    epsilon: float,
) -> dict[str, float]:
    model.eval()
    ratios: list[torch.Tensor] = []
    for inputs, targets, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs)
        predictions = normalizer.denormalize_output(predictions)
        targets = normalizer.denormalize_output(targets)
        ratios.append(relative_l1_per_sample(predictions, targets, epsilon).cpu())
    values = torch.cat(ratios).numpy() * 100.0
    return {
        "median_percent": float(np.median(values)),
        "mean_percent": float(np.mean(values)),
        "std_percent": float(np.std(values)),
    }


def _resume_training(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: MinMaxNormalizer,
    device: torch.device,
) -> tuple[int, float, int, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_val_relative_l1",
        "normalization",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise KeyError(f"resume checkpoint is missing keys: {missing}")
    checkpoint_normalizer = MinMaxNormalizer.from_state(state["normalization"])
    if checkpoint_normalizer != normalizer:
        raise ValueError("resume checkpoint normalization differs from config")
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return (
        int(state["epoch"]) + 1,
        float(state["best_val_relative_l1"]),
        int(state.get("bad_epochs", 0)),
        list(state.get("history", [])),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_config = config["training"]
    epochs = int(args.epochs if args.epochs is not None else train_config["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = select_device(args.device or str(train_config["device"]))
    seed = int(config["experiment"]["seed"])
    set_reproducibility(seed, bool(config["experiment"].get("deterministic", True)))
    normalizer = MinMaxNormalizer.from_config(config)

    source = data_file(config, "train_file")
    train_dataset = NavierStokesH5Dataset(
        source,
        numeric_sample_ids(config["data"]["train"]),
        normalizer,
        str(config["data"]["input_key"]),
        str(config["data"]["output_key"]),
    )
    validation_dataset = NavierStokesH5Dataset(
        source,
        numeric_sample_ids(config["data"]["validation"]),
        normalizer,
        str(config["data"]["input_key"]),
        str(config["data"]["output_key"]),
    )
    batch_size = int(train_config["batch_size"])
    workers = int(train_config["num_workers"])
    train_loader = _make_loader(train_dataset, batch_size, workers, True, seed, device)
    validation_loader = _make_loader(
        validation_dataset, batch_size, workers, False, seed, device
    )

    model = build_model(config["model"]).to(device)
    parameter_count = count_trainable_parameters(model)
    optimizer_name = str(train_config["optimizer"])
    if optimizer_name != "Adam":
        raise ValueError(f"paper reproduction requires Adam, got {optimizer_name}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    if str(train_config["scheduler"]) != "StepLR":
        raise ValueError("paper reproduction requires StepLR")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_config["scheduler_step_size"]),
        gamma=float(train_config["scheduler_gamma"]),
    )

    start_epoch = 1
    best_validation = float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        start_epoch, best_validation, bad_epochs, history = _resume_training(
            Path(args.resume).expanduser().resolve(),
            model,
            optimizer,
            scheduler,
            normalizer,
            device,
        )

    checkpoint_path = project_path(config["paths"]["checkpoint"])
    results_dir = project_path(config["paths"]["results_dir"])
    history_path = results_dir / "training_history.json"
    patience = int(train_config["early_stopping_patience"])
    log_interval = max(1, int(train_config["log_interval"]))

    print(
        f"experiment={config['experiment']['name']} device={device} "
        f"python={platform.python_version()} torch={torch.__version__}",
        flush=True,
    )
    print(
        f"train_samples={len(train_dataset)} val_samples={len(validation_dataset)} "
        f"batch_size={batch_size} parameters={parameter_count:,} "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for batch_index, (inputs, targets, _) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = F.l1_loss(predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite training loss at epoch={epoch}, batch={batch_index}: {loss}"
                )
            loss.backward()
            optimizer.step()
            batch_samples = inputs.shape[0]
            loss_sum += float(loss.detach()) * batch_samples
            sample_count += batch_samples
            if batch_index % log_interval == 0 or batch_index == len(train_loader):
                print(
                    f"train epoch={epoch}/{epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={float(loss.detach()):.8f} running_loss={loss_sum/sample_count:.8f}",
                    flush=True,
                )

        train_loss = loss_sum / sample_count
        validation = validate(
            model,
            validation_loader,
            normalizer,
            device,
            float(config["normalization"]["epsilon"]),
        )
        scheduler.step()
        improved = validation["median_percent"] < best_validation
        if improved:
            best_validation = validation["median_percent"]
            bad_epochs = 0
        else:
            bad_epochs += 1

        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_l1": train_loss,
            "validation_relative_l1_median_percent": validation["median_percent"],
            "validation_relative_l1_mean_percent": validation["mean_percent"],
            "validation_relative_l1_std_percent": validation["std_percent"],
            "best_validation_percent": best_validation,
        }
        history.append(record)
        print(
            f"eval epoch={epoch}/{epochs} lr={learning_rate:.8g} "
            f"train_l1={train_loss:.8f} "
            f"val_rel_l1_median={validation['median_percent']:.6f}% "
            f"val_rel_l1_mean={validation['mean_percent']:.6f}% "
            f"best={best_validation:.6f}% bad_epochs={bad_epochs}/{patience}",
            flush=True,
        )

        if improved:
            checkpoint = {
                "schema_version": "cno-navier-stokes-checkpoint-v1",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_relative_l1": best_validation,
                "bad_epochs": bad_epochs,
                "normalization": normalizer.state_dict(),
                "config": config,
                "seed": seed,
                "parameter_count": parameter_count,
                "history": history,
            }
            atomic_torch_save(checkpoint, checkpoint_path)
            print(
                f"checkpoint saved path={checkpoint_path} "
                f"val_rel_l1_median={best_validation:.6f}%",
                flush=True,
            )

        atomic_json_dump(
            {
                "experiment": config["experiment"],
                "device": str(device),
                "parameter_count": parameter_count,
                "normalization": normalizer.state_dict(),
                "best_validation_percent": best_validation,
                "history": history,
            },
            history_path,
        )
        if bad_epochs >= patience:
            print(
                f"early stopping at epoch={epoch}; no improvement for {patience} epochs",
                flush=True,
            )
            break

    print(
        f"training complete best_val_rel_l1_median={best_validation:.6f}% "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
