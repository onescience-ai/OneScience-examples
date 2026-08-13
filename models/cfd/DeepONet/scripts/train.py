#!/usr/bin/env python3
"""Train the paper-scale or explicitly reduced DeepONet experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.DeepONet import build_model, count_parameters  # noqa: E402
from models.dataset import OperatorDataset, build_datasets, resolve_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument(
        "--experiment",
        default="antiderivative",
        help="One experiment name or 'all' for all four paper experiments.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Variant name, 'all', or omit to use each experiment's paper default.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run reduced non-paper settings.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def load_config(path: Path, smoke_test: bool) -> Dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Configuration {path} must contain a mapping")
    return resolve_config(raw, smoke_test=smoke_test)


def resolve_path(path_value: str, root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else root / path


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but CUDA is unavailable")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: OperatorDataset,
    batch_size: int | None,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    effective_batch = len(dataset) if batch_size is None else int(batch_size)
    if effective_batch < 1:
        raise ValueError("batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def metric_values(prediction: np.ndarray, target: np.ndarray, trim_fraction: float = 0.0) -> Dict[str, float]:
    squared_error = np.square(prediction.astype(np.float64) - target.astype(np.float64)).reshape(-1)
    mse = float(np.mean(squared_error))
    denominator = float(np.linalg.norm(target.astype(np.float64).reshape(-1)))
    relative_l2 = float(
        np.linalg.norm(prediction.astype(np.float64).reshape(-1) - target.astype(np.float64).reshape(-1))
        / max(denominator, np.finfo(np.float64).eps)
    )
    metrics = {"test_mse": mse, "relative_l2": relative_l2}
    if trim_fraction > 0.0:
        remove_count = min(len(squared_error) - 1, int(np.ceil(len(squared_error) * trim_fraction)))
        kept = np.partition(squared_error, len(squared_error) - remove_count - 1)[
            : len(squared_error) - remove_count
        ]
        metrics["trimmed_test_mse"] = float(np.mean(kept))
    return metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: OperatorDataset,
    device: torch.device,
    batch_size: int,
    trim_fraction: float,
    num_workers: int,
) -> Dict[str, float]:
    loader = make_loader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions, targets = [], []
    model.eval()
    for branch, trunk, target in loader:
        output = model(branch.to(device), trunk.to(device))
        predictions.append(output.detach().cpu().numpy())
        targets.append(target.numpy())
    return metric_values(np.concatenate(predictions), np.concatenate(targets), trim_fraction)


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_bundle(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"format_version": 1, "entries": {}}
    bundle = torch_load(path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("entries"), dict):
        raise ValueError(f"Checkpoint {path} is not a DeepONet indexed bundle")
    return bundle


def save_bundle_entry(path: Path, key: str, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(path)
    bundle["format_version"] = 1
    bundle["entries"][key] = _cpu_copy(dict(entry))
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pth", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(bundle, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def append_history(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def variants_for(config: Mapping[str, Any], experiment: str, requested: str | None) -> Iterable[str]:
    if requested is None:
        return [str(config["experiments"][experiment]["default_variant"])]
    if requested == "all":
        return list(config["variants"])
    if requested not in config["variants"]:
        raise KeyError(f"Unknown variant {requested!r}; choose from {list(config['variants'])}")
    return [requested]


def run_training(
    config: Dict[str, Any],
    experiment: str,
    variant: str,
    *,
    no_cache: bool,
    resume: bool,
) -> None:
    experiment_config = config["experiments"][experiment]
    training_config = config["training_defaults"]
    device = select_device(str(config["runtime"]["device"]))
    seed = int(config["runtime"]["seed"])
    set_seed(seed)

    if no_cache:
        from models.dataset import build_split

        train_dataset = build_split(config, experiment, "train", PROJECT_ROOT, use_cache=False)
        test_dataset = build_split(config, experiment, "test", PROJECT_ROOT, use_cache=False)
    else:
        train_dataset, test_dataset = build_datasets(config, experiment, PROJECT_ROOT)

    model = build_model(config, experiment, variant).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss(reduction="mean")
    checkpoint_path = resolve_path(str(config["paths"]["checkpoint"]))
    entry_key = f"{experiment}/{variant}"
    start_iteration = 0
    best_metric = float("inf")
    if resume and checkpoint_path.exists():
        entry = load_bundle(checkpoint_path)["entries"].get(entry_key)
        if entry is None:
            raise KeyError(f"Cannot resume: {entry_key!r} is absent from {checkpoint_path}")
        model.load_state_dict(entry["model_state"], strict=True)
        optimizer.load_state_dict(entry["optimizer_state"])
        start_iteration = int(entry["iteration"])
        best_metric = float(entry["best_metric"])

    batch_size = training_config.get("batch_size")
    train_loader = make_loader(
        train_dataset,
        None if batch_size is None else int(batch_size),
        shuffle=True,
        num_workers=int(config["runtime"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    train_iterator = iter(train_loader)
    iterations = int(experiment_config["iterations"])
    print_every = int(training_config["print_every"])
    evaluate_every = int(training_config["evaluate_every"])
    eval_batch_size = int(training_config["evaluation_batch_size"])
    trim_fraction = float(experiment_config.get("trim_fraction", 0.0))
    history_path = resolve_path(str(config["paths"]["results"])) / experiment / variant / "history.csv"
    paper_scale = bool(config["project"]["paper_scale"])
    if not resume and history_path.exists():
        history_path.unlink()

    print(
        f"START experiment={experiment} variant={variant} device={device} "
        f"parameters={count_parameters(model)} train_points={len(train_dataset)} "
        f"test_points={len(test_dataset)} iterations={iterations} paper_scale={paper_scale}",
        flush=True,
    )
    for iteration in range(start_iteration + 1, iterations + 1):
        try:
            branch, trunk, target = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            branch, trunk, target = next(train_iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(branch.to(device), trunk.to(device))
        loss = criterion(prediction, target.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at iteration {iteration}")
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())

        should_evaluate = iteration == 1 or iteration % evaluate_every == 0 or iteration == iterations
        should_print = iteration == 1 or iteration % print_every == 0 or should_evaluate
        metrics: Dict[str, float] = {}
        if should_evaluate:
            metrics = evaluate(
                model,
                test_dataset,
                device,
                eval_batch_size,
                trim_fraction,
                int(config["runtime"].get("num_workers", 0)),
            )
            row: Dict[str, Any] = {
                "iteration": iteration,
                "train_loss": train_loss,
                "test_mse": metrics["test_mse"],
                "relative_l2": metrics["relative_l2"],
                "generalization_error": metrics["test_mse"] - train_loss,
                "paper_scale": paper_scale,
            }
            if "trimmed_test_mse" in metrics:
                row["trimmed_test_mse"] = metrics["trimmed_test_mse"]
            append_history(history_path, row)
            if metrics["test_mse"] < best_metric:
                best_metric = metrics["test_mse"]
                save_bundle_entry(
                    checkpoint_path,
                    entry_key,
                    {
                        "experiment": experiment,
                        "variant": variant,
                        "iteration": iteration,
                        "best_metric": best_metric,
                        "metric_name": "test_mse",
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "run_config": copy.deepcopy(config),
                        "paper_scale": paper_scale,
                    },
                )
        if should_print:
            metric_text = " ".join(f"{name}={value:.8e}" for name, value in metrics.items())
            print(
                f"TRAIN experiment={experiment} variant={variant} iteration={iteration}/{iterations} "
                f"train_loss={train_loss:.8e} {metric_text}".rstrip(),
                flush=True,
            )
    print(
        f"DONE experiment={experiment} variant={variant} best_test_mse={best_metric:.8e} "
        f"checkpoint={checkpoint_path} history={history_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.smoke_test)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.seed is not None:
        config["runtime"]["seed"] = args.seed
    experiments = list(config["experiments"]) if args.experiment == "all" else [args.experiment]
    unknown = [name for name in experiments if name not in config["experiments"]]
    if unknown:
        raise KeyError(f"Unknown experiments: {unknown}; choose from {list(config['experiments'])}")
    for experiment in experiments:
        for variant in variants_for(config, experiment, args.variant):
            run_training(
                config,
                experiment,
                variant,
                no_cache=args.no_cache,
                resume=args.resume or bool(config["training_defaults"].get("resume", False)),
            )


if __name__ == "__main__":
    main()
