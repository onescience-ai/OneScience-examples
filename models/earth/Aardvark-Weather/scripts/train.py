"""Train the official Aardvark Day-1 TAS model on official-schema tasks."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.aardvark_adapter import build_one_day_model
from model.sample_dataset import AardvarkTaskDataset, collate_tasks, discover_samples, split_samples


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=ROOT / "conf" / "config.yaml")
    known, _ = pre_parser.parse_known_args()
    config = yaml.safe_load(known.config.read_text())["training"]

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument("--data", type=Path, default=ROOT / config["data"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / config["output_dir"])
    parser.add_argument("--epochs", type=int, default=config["epochs"])
    parser.add_argument("--train-steps", type=int, default=config["train_steps"])
    parser.add_argument("--validation-steps", type=int, default=config["validation_steps"])
    parser.add_argument("--validation-fraction", type=float, default=config["validation_fraction"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=config["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=config["weight_decay"])
    parser.add_argument("--gradient-clip", type=float, default=config["gradient_clip"])
    parser.add_argument("--patience", type=int, default=config["patience"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--train-modules", choices=("decoder", "all"), default=config["train_modules"])
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def masked_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    valid = torch.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        raise RuntimeError("Aardvark task has no finite station targets")
    error = prediction[valid] - target[valid]
    return torch.sqrt(error.square().mean()), error.abs().mean(), count


def configure_trainable_parameters(model: torch.nn.Module, train_modules: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(train_modules == "all")
    if train_modules == "decoder":
        for parameter in model.sf_model.parameters():
            parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters selected")
    return parameters


def checkpoint_payload(model, optimizer, scheduler, args, epoch, best_validation_rmse, history):
    model_state = model.state_dict() if args.train_modules == "all" else model.sf_model.state_dict()
    return {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_rmse": best_validation_rmse,
        "history": history,
        "train_modules": args.train_modules,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The official Aardvark model requires CUDA")
    if min(args.epochs, args.train_steps, args.validation_steps, args.batch_size) < 1:
        raise ValueError("epochs, steps and batch_size must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(args.data)
    train_samples, validation_samples = split_samples(samples, args.validation_fraction, args.seed)
    train_loader = DataLoader(
        AardvarkTaskDataset(train_samples, args.train_steps * args.batch_size),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_tasks,
    )
    validation_loader = DataLoader(
        AardvarkTaskDataset(validation_samples, args.validation_steps * args.batch_size),
        batch_size=args.batch_size,
        collate_fn=collate_tasks,
    )

    model = build_one_day_model(ROOT / "weights", ROOT / "official-src", "cuda")
    model.return_gridded = False
    parameters = configure_trainable_parameters(model, args.train_modules)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )

    start_epoch = 0
    best_validation_rmse = math.inf
    history: list[dict[str, float | int]] = []
    if args.resume:
        payload = torch.load(args.resume, map_location="cuda", weights_only=False)
        if payload["train_modules"] != args.train_modules:
            raise ValueError("--train-modules must match the resumed checkpoint")
        target_model = model if args.train_modules == "all" else model.sf_model
        target_model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation_rmse = float(payload["best_validation_rmse"])
        history = payload["history"]
        torch.set_rng_state(payload["torch_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])

    epochs_without_improvement = 0
    last_path = args.output_dir / "last.pth"
    best_path = args.output_dir / "best.pth"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.train_modules == "decoder":
            model.se_model.eval()
            model.forecast_model.eval()
            model.sf_model.train()
        train_rmse = []
        train_mae = []
        for task in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(task)
            target = task["y_target"].to(prediction.device)
            rmse, mae, _ = masked_metrics(prediction, target)
            rmse.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()
            train_rmse.append(float(rmse.detach()))
            train_mae.append(float(mae.detach()))

        model.eval()
        validation_rmse = []
        validation_mae = []
        valid_stations = 0
        with torch.inference_mode():
            for task in validation_loader:
                prediction = model(task)
                rmse, mae, valid_stations = masked_metrics(prediction, task["y_target"].to(prediction.device))
                validation_rmse.append(float(rmse))
                validation_mae.append(float(mae))

        record = {
            "epoch": epoch,
            "train_rmse": float(np.mean(train_rmse)),
            "train_mae": float(np.mean(train_mae)),
            "validation_rmse": float(np.mean(validation_rmse)),
            "validation_mae": float(np.mean(validation_mae)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "valid_stations": valid_stations,
        }
        history.append(record)
        scheduler.step(record["validation_rmse"])
        improved = record["validation_rmse"] < best_validation_rmse
        if improved:
            best_validation_rmse = record["validation_rmse"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = checkpoint_payload(model, optimizer, scheduler, args, epoch, best_validation_rmse, history)
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record, sort_keys=True))
        if epochs_without_improvement >= args.patience:
            break

    report = {
        "status": "completed",
        "epochs_completed": len(history),
        "train_modules": args.train_modules,
        "train_samples": [str(path) for path in train_samples],
        "validation_samples": [str(path) for path in validation_samples],
        "best_validation_rmse": best_validation_rmse,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    (args.output_dir / "train.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
