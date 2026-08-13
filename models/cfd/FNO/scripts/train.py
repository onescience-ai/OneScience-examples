#!/usr/bin/env python3
"""Train the paper-specified recurrent FNO-2D Navier--Stokes model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.io
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FNO-2D on the validated Navier-Stokes trajectory MAT file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="YAML configuration path.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke only).")
    parser.add_argument(
        "--rollout-steps", type=int, default=None, help="Override rollout steps (smoke only)."
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Limit train samples (smoke only)."
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=None, help="Limit test samples (smoke only)."
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or an explicit torch device."
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Output checkpoint path.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a formal training-state checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--run-type", choices=("formal", "smoke"), default="formal", help="Run provenance tag."
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("paper", "data", "model", "training", "paths"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    data = config["data"]
    model = config["model"]
    training = config["training"]
    required_data = {
        "root",
        "file",
        "key",
        "layout",
        "dtype",
        "expected_shape",
        "resolution",
        "ntrain",
        "ntest",
        "test_start",
        "history",
        "horizon",
        "normalization",
    }
    required_model = {
        "input_channels",
        "output_channels",
        "width",
        "modes1",
        "modes2",
        "num_layers",
        "projection_width",
        "use_grid",
    }
    required_training = {
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "scheduler_step_size",
        "scheduler_gamma",
        "seed",
        "dtype",
        "relative_l2_epsilon",
        "checkpoint_monitor",
    }
    for label, mapping, required in (
        ("data", data, required_data),
        ("model", model, required_model),
        ("training", training, required_training),
    ):
        missing = sorted(required.difference(mapping))
        if missing:
            raise ValueError(f"Missing {label} configuration keys: {missing}")

    if data["layout"] != "N,H,W,T":
        raise ValueError("This reproduction requires data.layout=N,H,W,T")
    if data["dtype"] != "float32" or training["dtype"] != "float32":
        raise ValueError("The audited reproduction requires float32 data and training")
    if data["normalization"] != "none":
        raise ValueError("The paper-faithful default requires normalization=none")
    if int(data["history"]) != 10 or int(model["input_channels"]) != 10:
        raise ValueError("The FNO-2D experiment requires ten history channels")
    if int(data["horizon"]) != 10 or int(model["output_channels"]) != 1:
        raise ValueError("The experiment requires a ten-step rollout of one-step outputs")
    if int(model["width"]) != 32 or int(model["num_layers"]) != 4:
        raise ValueError("Strict paper settings require width=32 and num_layers=4")
    if int(model["modes1"]) != 12 or int(model["modes2"]) != 12:
        raise ValueError("Strict paper settings require 12 retained modes per axis")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("The paper specifies Adam")
    if str(training["scheduler"]).lower() != "step_lr":
        raise ValueError("The paper schedule is represented by StepLR")
    if str(training["checkpoint_monitor"]) != "train_full_relative_l2":
        raise ValueError("Test metrics must not select the checkpoint")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def data_file_from_config(config: dict[str, Any]) -> Path:
    data = config["data"]
    path = Path(str(data["root"])).expanduser() / str(data["file"])
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Navier-Stokes MAT file does not exist: {path}")
    return path


def load_trajectory_array(config: dict[str, Any]) -> np.ndarray:
    data = config["data"]
    path = data_file_from_config(config)
    key = str(data["key"])
    payload = scipy.io.loadmat(path, variable_names=[key])
    if key not in payload:
        raise KeyError(f"MAT field {key!r} is missing from {path}")
    trajectories = payload[key]
    expected_shape = tuple(int(value) for value in data["expected_shape"])
    if trajectories.shape != expected_shape:
        raise ValueError(
            f"Expected {key} shape {expected_shape}, received {trajectories.shape}"
        )
    if trajectories.dtype != np.float32:
        raise TypeError(f"Expected {key} dtype float32, received {trajectories.dtype}")
    if not np.isfinite(trajectories).all():
        raise ValueError("Trajectory array contains NaN or Inf")
    return trajectories


def build_datasets(
    config: dict[str, Any],
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    rollout_steps: int | None = None,
) -> tuple[TensorDataset, TensorDataset]:
    data = config["data"]
    trajectories = load_trajectory_array(config)
    history = int(data["history"])
    horizon = int(data["horizon"])
    steps = horizon if rollout_steps is None else int(rollout_steps)
    if not 1 <= steps <= horizon:
        raise ValueError(f"rollout_steps must be in [1,{horizon}], got {steps}")

    ntrain = int(data["ntrain"])
    ntest = int(data["ntest"])
    train_start = int(data.get("train_start", 0))
    test_start = int(data["test_start"])
    train_count = ntrain if max_train_samples is None else min(ntrain, max_train_samples)
    test_count = ntest if max_test_samples is None else min(ntest, max_test_samples)
    if train_count <= 0 or test_count <= 0:
        raise ValueError("Training and test sample counts must be positive")

    train = trajectories[train_start : train_start + train_count]
    test = trajectories[test_start : test_start + test_count]
    train_history = torch.from_numpy(train[..., :history])
    train_target = torch.from_numpy(train[..., history : history + steps])
    test_history = torch.from_numpy(test[..., :history])
    test_target = torch.from_numpy(test[..., history : history + steps])
    expected_hw = tuple(int(value) for value in data["resolution"])
    expected_history_shape = (expected_hw[0], expected_hw[1], history)
    if tuple(train_history.shape[1:]) != expected_history_shape:
        raise ValueError(f"Invalid train history shape: {train_history.shape}")
    if tuple(test_history.shape[1:]) != expected_history_shape:
        raise ValueError(f"Invalid test history shape: {test_history.shape}")
    return TensorDataset(train_history, train_target), TensorDataset(test_history, test_target)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but torch reports no CUDA/DCU device")
    return device


def relative_l2_per_sample(prediction: Tensor, target: Tensor, epsilon: float) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Relative L2 shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.ndim < 2:
        raise ValueError("Relative L2 inputs must include a batch and at least one feature axis")
    difference = torch.linalg.vector_norm(
        (prediction - target).reshape(prediction.shape[0], -1), dim=1
    )
    denominator = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1)
    return difference / (denominator + epsilon)


def autoregressive_rollout(
    model: nn.Module,
    history: Tensor,
    target: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if history.ndim != 4 or target.ndim != 4:
        raise ValueError("history and target must be channel-last four-dimensional tensors")
    window = history
    predictions: list[Tensor] = []
    step_ratios: list[Tensor] = []
    for step in range(target.shape[-1]):
        prediction = model(window)
        expected_step_shape = (*history.shape[:-1], 1)
        if tuple(prediction.shape) != expected_step_shape:
            raise ValueError(
                f"Model returned {tuple(prediction.shape)}, expected {expected_step_shape}"
            )
        target_step = target[..., step : step + 1]
        predictions.append(prediction)
        step_ratios.append(relative_l2_per_sample(prediction, target_step, epsilon))
        window = torch.cat((window[..., 1:], prediction), dim=-1)
    rollout = torch.cat(predictions, dim=-1)
    ratios = torch.stack(step_ratios, dim=1)
    backward_loss = ratios.mean(dim=0).sum()
    return rollout, ratios, backward_loss


def build_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_samples = 0
    total_step_ratio = 0.0
    total_full_ratio = 0.0
    steps_per_sample: int | None = None

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for history, target in loader:
            history = history.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction, step_ratios, backward_loss = autoregressive_rollout(
                model, history, target, epsilon
            )
            full_ratios = relative_l2_per_sample(prediction, target, epsilon)
            if not torch.isfinite(backward_loss) or not torch.isfinite(full_ratios).all():
                raise FloatingPointError("Non-finite training/evaluation loss encountered")
            if training:
                backward_loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Non-finite gradient in parameter {name}")
                optimizer.step()

            batch_samples = int(history.shape[0])
            total_samples += batch_samples
            total_step_ratio += float(step_ratios.detach().sum().cpu())
            total_full_ratio += float(full_ratios.detach().sum().cpu())
            steps_per_sample = int(step_ratios.shape[1])

    if total_samples == 0 or steps_per_sample is None:
        raise RuntimeError("DataLoader produced no batches")
    return {
        "step_loss_sum": total_step_ratio / total_samples,
        "mean_step_relative_l2": total_step_ratio / (total_samples * steps_per_sample),
        "full_relative_l2": total_full_ratio / total_samples,
        "samples": float(total_samples),
        "rollout_steps": float(steps_per_sample),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    device_name = "cpu"
    if device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
        "device": str(device),
        "device_name": device_name,
        "torch_cuda_version": torch.version.cuda,
        "hostname": platform.node(),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Resume checkpoint root must be a mapping")
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_train_full_relative_l2",
        "monitor",
        "test_selected",
        "config",
        "run_type",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"Resume checkpoint is missing required keys: {missing}")
    if checkpoint["run_type"] != "formal":
        raise ValueError("Formal training can only resume a formal checkpoint")
    if checkpoint["monitor"] != "train_full_relative_l2":
        raise ValueError(f"Unexpected checkpoint monitor: {checkpoint['monitor']}")
    if checkpoint["test_selected"] is not False:
        raise ValueError("This reproduction forbids resuming a test-selected checkpoint")
    return checkpoint


def capture_rng_state(train_loader: DataLoader) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_loader_generator": train_loader.generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], train_loader: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    train_loader.generator.set_state(state["train_loader_generator"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed, bool(training.get("deterministic", True)))
    device = select_device(args.device)

    epochs = int(training["epochs"] if args.epochs is None else args.epochs)
    batch_size = int(training["batch_size"] if args.batch_size is None else args.batch_size)
    horizon = int(config["data"]["horizon"])
    rollout_steps = horizon if args.rollout_steps is None else int(args.rollout_steps)
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.run_type == "formal" and any(
        value is not None
        for value in (
            args.epochs,
            args.rollout_steps,
            args.max_train_samples,
            args.max_test_samples,
            args.output_dir,
            args.checkpoint,
        )
    ):
        raise ValueError(
            "Formal runs use the exact configured data, epochs, rollout, checkpoint, "
            "and results paths; overrides require --run-type smoke"
        )

    output_dir = (
        resolve_project_path(config["paths"]["results_dir"])
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    checkpoint_path = (
        resolve_project_path(config["paths"]["checkpoint"])
        if args.checkpoint is None
        else args.checkpoint.expanduser().resolve()
    )
    latest_checkpoint_path = checkpoint_path.with_name("last_model.pth")
    history_path = (
        resolve_project_path(config["paths"]["train_history"])
        if args.output_dir is None
        else output_dir / "train_history.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = build_datasets(
        config,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        rollout_steps=rollout_steps,
    )
    pin_memory = bool(training.get("pin_memory", True)) and device.type == "cuda"
    train_loader = build_loader(
        train_dataset,
        batch_size,
        True,
        int(training.get("num_workers", 0)),
        pin_memory,
        seed,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size,
        False,
        int(training.get("num_workers", 0)),
        pin_memory,
        seed,
    )

    # Move devices without forcing a global dtype conversion: the pointwise
    # parameters are float32 while spectral weights must remain complex64.
    model = build_model_from_config(config).to(device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    paper_parameter_count = int(config["paper"]["reference_parameter_count"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["scheduler_step_size"]),
        gamma=float(training["scheduler_gamma"]),
    )
    epsilon = float(training["relative_l2_epsilon"])

    run_started = datetime.now(timezone.utc).isoformat()
    history_payload: dict[str, Any] = {
        "schema_version": "fno-ns2d-train-history-v1",
        "run_type": args.run_type,
        "started_at": run_started,
        "completed_at": None,
        "config_path": str(args.config.expanduser().resolve()),
        "data_path": str(data_file_from_config(config)),
        "checkpoint_path": str(checkpoint_path),
        "config": deepcopy(config),
        "environment": environment_metadata(device),
        "parameter_count": parameter_count,
        "paper_parameter_count": paper_parameter_count,
        "parameter_count_difference": parameter_count - paper_parameter_count,
        "test_selected": False,
        "epochs_requested": epochs,
        "rollout_steps": rollout_steps,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "history": [],
    }

    best_metric = float("inf")
    start_epoch = 1
    resume_exact_rng = True
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        resume_checkpoint = load_training_checkpoint(resume_path, device)
        if resume_checkpoint["config"] != config:
            raise ValueError("Resume checkpoint configuration differs from the current YAML")
        resume_epoch = int(resume_checkpoint["epoch"])
        if not 1 <= resume_epoch < epochs:
            raise ValueError(
                f"Resume epoch must be in [1,{epochs - 1}], received {resume_epoch}"
            )
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Training history required for audited resume is missing: {history_path}"
            )
        with history_path.open("r", encoding="utf-8") as handle:
            previous_history = json.load(handle)
        if not isinstance(previous_history, dict):
            raise TypeError("Existing training history root must be a mapping")
        records = previous_history.get("history")
        if not isinstance(records, list) or len(records) < resume_epoch:
            raise ValueError(
                f"Existing history has {len(records) if isinstance(records, list) else 0} "
                f"records, fewer than resume epoch {resume_epoch}"
            )
        retained_records = records[:resume_epoch]
        if int(retained_records[-1]["epoch"]) != resume_epoch:
            raise ValueError("Existing training history is not contiguous at the resume epoch")
        checkpoint_metric = float(resume_checkpoint["best_train_full_relative_l2"])
        history_best = min(float(record["train_full_relative_l2"]) for record in retained_records)
        if not np.isclose(checkpoint_metric, history_best, rtol=1e-7, atol=1e-9):
            raise ValueError(
                f"Resume checkpoint best metric {checkpoint_metric} differs from retained "
                f"history best {history_best}"
            )

        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if not bool(resume_checkpoint.get("scheduler_step_applied", False)):
            scheduler.step()
        rng_state = resume_checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            restore_rng_state(rng_state, train_loader)
        else:
            resume_exact_rng = False

        discarded_records = len(records) - resume_epoch
        history_payload = previous_history
        history_payload["history"] = retained_records
        history_payload["completed_at"] = None
        history_payload["completed"] = False
        history_payload["epochs_requested"] = epochs
        history_payload["resume_exact_rng"] = resume_exact_rng
        resume_events = history_payload.setdefault("resume_events", [])
        resume_events.append(
            {
                "resumed_at": run_started,
                "checkpoint_path": str(resume_path),
                "checkpoint_epoch": resume_epoch,
                "discarded_history_records": discarded_records,
                "exact_rng_state_restored": resume_exact_rng,
            }
        )
        best_metric = checkpoint_metric
        start_epoch = resume_epoch + 1
        atomic_write_json(history_path, history_payload)
        print(
            f"resume checkpoint={resume_path} epoch={resume_epoch} "
            f"discarded_history_records={discarded_records} "
            f"exact_rng_state_restored={'yes' if resume_exact_rng else 'no'}",
            flush=True,
        )
    print(
        f"device={device} parameters={parameter_count} "
        f"paper_parameters={paper_parameter_count} delta={parameter_count-paper_parameter_count}",
        flush=True,
    )

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs + 1):
        synchronize(device)
        epoch_start = time.perf_counter()
        lr_used = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(model, train_loader, device, epsilon, optimizer)
        test_metrics = run_epoch(model, test_loader, device, epsilon, optimizer=None)
        synchronize(device)
        duration = time.perf_counter() - epoch_start
        peak_memory_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" and torch.cuda.is_available()
            else 0
        )

        monitor = float(train_metrics["full_relative_l2"])
        is_best = monitor < best_metric
        if is_best:
            best_metric = monitor
            checkpoint = {
                "schema_version": "fno-ns2d-checkpoint-v1",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_train_full_relative_l2": best_metric,
                "test_full_relative_l2_at_best_epoch": float(
                    test_metrics["full_relative_l2"]
                ),
                "monitor": "train_full_relative_l2",
                "test_selected": False,
                "config": deepcopy(config),
                "seed": seed,
                "parameter_count": parameter_count,
                "paper_parameter_count": paper_parameter_count,
                "parameter_count_difference": parameter_count - paper_parameter_count,
                "run_type": args.run_type,
                "scheduler_step_applied": False,
                "rng_state": capture_rng_state(train_loader),
            }
            atomic_torch_save(checkpoint_path, checkpoint)

        epoch_record = {
            "epoch": epoch,
            "learning_rate": lr_used,
            "duration_seconds": duration,
            "train_step_loss_sum": float(train_metrics["step_loss_sum"]),
            "train_mean_step_relative_l2": float(
                train_metrics["mean_step_relative_l2"]
            ),
            "train_full_relative_l2": float(train_metrics["full_relative_l2"]),
            "test_mean_step_relative_l2": float(test_metrics["mean_step_relative_l2"]),
            "test_full_relative_l2": float(test_metrics["full_relative_l2"]),
            "peak_accelerator_memory_bytes": peak_memory_bytes,
            "best": is_best,
        }
        history_payload["history"].append(epoch_record)
        history_payload["best_epoch"] = next(
            record["epoch"]
            for record in reversed(history_payload["history"])
            if record["best"]
        )
        history_payload["best_train_full_relative_l2"] = best_metric
        atomic_write_json(history_path, history_payload)
        print(
            f"epoch={epoch:04d}/{epochs:04d} time={duration:.3f}s lr={lr_used:.6g} "
            f"train_step_loss_sum={train_metrics['step_loss_sum']:.8f} "
            f"train_step_rel_l2={train_metrics['mean_step_relative_l2']:.8f} "
            f"train_full_rel_l2={train_metrics['full_relative_l2']:.8f} "
            f"test_step_rel_l2={test_metrics['mean_step_relative_l2']:.8f} "
            f"test_full_rel_l2={test_metrics['full_relative_l2']:.8f} "
            f"peak_mem_gib={peak_memory_bytes / (1024 ** 3):.3f} "
            f"best={'yes' if is_best else 'no'}",
            flush=True,
        )
        scheduler.step()
        latest_checkpoint = {
            "schema_version": "fno-ns2d-training-state-v1",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_train_full_relative_l2": best_metric,
            "latest_train_full_relative_l2": float(train_metrics["full_relative_l2"]),
            "latest_test_full_relative_l2": float(test_metrics["full_relative_l2"]),
            "monitor": "train_full_relative_l2",
            "test_selected": False,
            "config": deepcopy(config),
            "seed": seed,
            "parameter_count": parameter_count,
            "paper_parameter_count": paper_parameter_count,
            "parameter_count_difference": parameter_count - paper_parameter_count,
            "run_type": args.run_type,
            "scheduler_step_applied": True,
            "rng_state": capture_rng_state(train_loader),
        }
        atomic_torch_save(latest_checkpoint_path, latest_checkpoint)

    history_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    history_payload["completed"] = True
    atomic_write_json(history_path, history_payload)
    print(
        f"training_complete best_epoch={history_payload['best_epoch']} "
        f"best_train_full_rel_l2={best_metric:.8f} checkpoint={checkpoint_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
