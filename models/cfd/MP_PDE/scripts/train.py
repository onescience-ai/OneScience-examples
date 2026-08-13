"""Train the paper-priority MP-PDE E3 solver."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataset import E3Dataset, generate_e3_hdf5  # noqa: E402
from models.pde import MPPDESolver  # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def build_model(config: Mapping[str, Any]) -> MPPDESolver:
    model_cfg, data_cfg = config["model"], config["data"]
    decoder, scaling = model_cfg["decoder"], model_cfg["scaling"]
    parameter_maxima = [data_cfg["equation"][name][1] for name in ("alpha_range", "beta_range", "gamma_range")]
    if model_cfg["aggregation"] != "sum":
        raise ValueError("Paper Eq. (9) requires model.aggregation=sum")
    if scaling["solution"] != "none":
        raise ValueError("The paper does not specify solution normalization; model.scaling.solution must be none")
    return MPPDESolver(
        time_window=int(model_cfg["time_window"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        message_passing_layers=int(model_cfg["message_passing_layers"]),
        neighbor_offsets=model_cfg["neighbor_offsets"],
        domain_length=float(data_cfg["domain_length"]),
        final_time=float(data_cfg["final_time"]),
        parameter_maxima=parameter_maxima,
        scale_coordinates=bool(scaling["coordinates"]),
        scale_parameters=bool(scaling["parameters"]),
        instance_norm_affine=bool(model_cfg["instance_norm_affine"]),
        decoder_middle_channels=int(decoder["middle_channels"]),
        decoder_kernels=decoder["kernel_sizes"],
        decoder_strides=decoder["strides"],
    )


def rmse_loss(prediction: Tensor, target: Tensor, epsilon: float) -> Tensor:
    return torch.sqrt(torch.mean((prediction - target) ** 2) + epsilon)


def _extract_windows(trajectories: Tensor, starts: Tensor, window: int, no_grad_pushes: int) -> Tuple[Tensor, Tensor]:
    histories, targets = [], []
    target_offset = window * (no_grad_pushes + 1)
    for sample, start_tensor in zip(trajectories, starts):
        start = int(start_tensor)
        histories.append(sample[start : start + window].transpose(0, 1))
        targets.append(sample[start + target_offset : start + target_offset + window].transpose(0, 1))
    return torch.stack(histories), torch.stack(targets)


def train_step(
    batch: Mapping[str, Tensor],
    model: MPPDESolver,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Mapping[str, Any],
    window_generator: torch.Generator,
) -> Tuple[float, int]:
    training = config["training"]
    window = int(config["model"]["time_window"])
    max_pushes = int(training["max_no_grad_pushes"])
    no_grad_pushes = 0 if epoch == 0 else int(torch.randint(0, max_pushes + 1, (1,), generator=window_generator))
    trajectories = batch["u"]
    num_times = trajectories.shape[1]
    max_start = num_times - window * (no_grad_pushes + 2)
    if max_start < 0:
        raise ValueError(f"nt={num_times} is too short for K={window} and pushes={no_grad_pushes}")
    starts = torch.randint(0, max_start + 1, (trajectories.shape[0],), generator=window_generator)
    history, target = _extract_windows(trajectories, starts, window, no_grad_pushes)
    history, target = history.to(device), target.to(device)
    x, parameters = batch["x"].to(device), batch["params"].to(device)
    times = batch["t"]
    dt = (times[:, 1] - times[:, 0]).to(device)
    current_time = torch.stack([times[index, int(start) + window - 1] for index, start in enumerate(starts)]).to(device)
    for _ in range(no_grad_pushes):
        with torch.no_grad():
            history = model(history, x, current_time, parameters, dt).detach()
        current_time = current_time + window * dt
    optimizer.zero_grad(set_to_none=True)
    prediction = model(history, x, current_time, parameters, dt)
    loss = rmse_loss(prediction, target, float(training["loss_epsilon"]))
    loss.backward()
    clip = training.get("gradient_clip")
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
    optimizer.step()
    return float(loss.detach().cpu()), no_grad_pushes


@torch.inference_mode()
def evaluate_bundle_rmse(
    model: MPPDESolver, loader: Iterable[Mapping[str, Tensor]], device: torch.device, window: int, epsilon: float
) -> float:
    model.eval()
    squared_error, count = 0.0, 0
    for batch in loader:
        history = batch["u"][:, :window].transpose(1, 2).to(device)
        target = batch["u"][:, window : 2 * window].transpose(1, 2).to(device)
        x, times, parameters = batch["x"].to(device), batch["t"], batch["params"].to(device)
        dt = (times[:, 1] - times[:, 0]).to(device)
        current_time = times[:, window - 1].to(device)
        prediction = model(history, x, current_time, parameters, dt)
        squared_error += float(torch.sum((prediction - target) ** 2).cpu())
        count += prediction.numel()
    return float(np.sqrt(squared_error / max(count, 1) + epsilon))


@torch.inference_mode()
def rollout_batch(model: MPPDESolver, batch: Mapping[str, Tensor], device: torch.device, window: int) -> Tensor:
    target = batch["u"].to(device)
    x, parameters = batch["x"].to(device), batch["params"].to(device)
    times = batch["t"].to(device)
    dt = times[:, 1] - times[:, 0]
    total_times = target.shape[1]
    prediction = torch.empty_like(target)
    prediction[:, :window] = target[:, :window]
    for start in range(window, total_times, window):
        history = prediction[:, start - window : start].transpose(1, 2)
        current_time = times[:, start - 1]
        bundle = model(history, x, current_time, parameters, dt).transpose(1, 2)
        length = min(window, total_times - start)
        prediction[:, start : start + length] = bundle[:, :length]
        if not torch.all(torch.isfinite(bundle)):
            raise FloatingPointError(f"Non-finite rollout output at forecast index {start}")
    return prediction


@torch.inference_mode()
def evaluate_rollout(
    model: MPPDESolver,
    loader: Iterable[Mapping[str, Tensor]],
    device: torch.device,
    window: int,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    per_time_sse: Optional[Tensor] = None
    element_count_per_time = 0
    total_samples = 0
    for batch in loader:
        if max_samples is not None:
            remaining = int(max_samples) - total_samples
            if remaining <= 0:
                break
            if batch["u"].shape[0] > remaining:
                batch = {key: value[:remaining] for key, value in batch.items()}
        prediction = rollout_batch(model, batch, device, window)
        target = batch["u"].to(device)
        error = prediction[:, window:] - target[:, window:]
        current = torch.sum(error.double() ** 2, dim=(0, 2)).cpu()
        per_time_sse = current if per_time_sse is None else per_time_sse + current
        element_count_per_time += error.shape[0] * error.shape[2]
        total_samples += error.shape[0]
    if per_time_sse is None or total_samples == 0:
        raise RuntimeError("No samples were available for rollout evaluation")
    per_time_mse = per_time_sse / element_count_per_time
    return {
        "accumulated_mse": float(per_time_mse.sum()),
        "rmse": float(torch.sqrt(per_time_mse.mean())),
        "samples": float(total_samples),
    }


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MP-PDE E3 model")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument("--generate-data", action="store_true", help="Generate configured data first when it is absent")
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--epochs", type=int, help="Explicit run override; does not modify config.yaml")
    args = parser.parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume and --init-checkpoint are mutually exclusive")

    config = load_config(args.config.resolve())
    training = config["training"]
    if training["optimizer"] != "AdamW" or training["loss"] != "rmse":
        raise ValueError("Paper-priority E3 requires AdamW and RMSE")
    data_path = project_path(config["paths"]["data"])
    if args.generate_data and (not data_path.exists() or args.overwrite_data):
        generate_e3_hdf5(config, data_path, overwrite=args.overwrite_data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset missing: {data_path}. Run with --generate-data or invoke models/dataset.py")

    seed = int(training["seed"])
    set_seed(seed, bool(training["deterministic"]))
    device = choose_device(args.device or str(training["device"]))
    window = int(config["model"]["time_window"])
    expected_nt, expected_nx = int(config["data"]["num_time_points"]), int(config["data"]["resolution"])
    datasets = {split: E3Dataset(data_path, split, expected_nt, expected_nx) for split in ("train", "valid", "test")}
    loader_generator = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(batch_size=int(training["batch_size"]), num_workers=int(training["num_workers"]), pin_memory=device.type == "cuda")
    train_loader = DataLoader(datasets["train"], shuffle=True, generator=loader_generator, **loader_kwargs)
    valid_loader = DataLoader(datasets["valid"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)
    window_generator = torch.Generator().manual_seed(seed + 104729)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler_cfg = training["scheduler"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(value) for value in scheduler_cfg["milestones"]], gamma=float(scheduler_cfg["gamma"])
    ) if scheduler_cfg["enabled"] else None
    start_epoch, best_metric, history = 0, float("inf"), []

    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=True)
        print(f"Initialized model weights from {args.init_checkpoint}; optimizer and scheduler were reset", flush=True)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        required = {
            "model_state", "optimizer_state", "epoch", "best_validation_accumulated_mse", "rng_state",
            "loader_generator_state", "window_generator_state",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"Resume checkpoint lacks training state: {sorted(missing)}")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None:
            if checkpoint.get("scheduler_state") is None:
                raise KeyError("Resume checkpoint lacks scheduler_state")
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_validation_accumulated_mse"])
        history = list(checkpoint.get("history", []))
        _restore_rng_state(checkpoint["rng_state"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        window_generator.set_state(checkpoint["window_generator_state"])
        print(f"Resumed training from {args.resume} at epoch {start_epoch}", flush=True)

    epochs = int(args.epochs or training["epochs"])
    if start_epoch >= epochs:
        raise ValueError(f"Resume epoch {start_epoch} is not below requested epochs={epochs}")
    checkpoint_path = project_path(config["paths"]["checkpoint"])
    history_path = project_path(config["paths"]["train_history"])
    print(f"Training MP-PDE E3 on device={device}, parameters={sum(p.numel() for p in model.parameters())}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        step_losses = []
        for step, batch in enumerate(train_loader, start=1):
            loss, pushes = train_step(batch, model, optimizer, device, epoch, config, window_generator)
            step_losses.append(loss)
            if step == 1 or step % int(training["print_interval"]) == 0 or step == len(train_loader):
                learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch + 1}/{epochs} step={step}/{len(train_loader)} pushes={pushes} "
                    f"loss_rmse={loss:.8e} lr={learning_rate:.3e}", flush=True,
                )
        validation_rmse = evaluate_bundle_rmse(model, valid_loader, device, window, float(training["loss_epsilon"]))
        validation_rollout = evaluate_rollout(model, valid_loader, device, window, training.get("validation_rollout_samples"))
        train_rmse = float(np.mean(step_losses))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "validation_bundle_rmse": validation_rmse,
            "validation_accumulated_mse": validation_rollout["accumulated_mse"],
            "learning_rate": learning_rate,
        }
        print(
            f"epoch={epoch + 1}/{epochs} train_rmse={train_rmse:.8e} val_rmse={validation_rmse:.8e} "
            f"val_accumulated_mse={validation_rollout['accumulated_mse']:.8e} lr={learning_rate:.3e}", flush=True,
        )
        improved = validation_rollout["accumulated_mse"] < best_metric
        if scheduler is not None:
            scheduler.step()
        if improved:
            best_metric = validation_rollout["accumulated_mse"]
            test_rollout = evaluate_rollout(model, test_loader, device, window, training.get("test_rollout_samples"))
            epoch_record["test_accumulated_mse_on_new_best"] = test_rollout["accumulated_mse"]
            epoch_record["test_rmse_on_new_best"] = test_rollout["rmse"]
            print(
                f"new_best epoch={epoch + 1} val_accumulated_mse={best_metric:.8e} "
                f"test_accumulated_mse={test_rollout['accumulated_mse']:.8e} test_rmse={test_rollout['rmse']:.8e}",
                flush=True,
            )
        history.append(epoch_record)
        atomic_json_dump(history, history_path)
        if improved:
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_validation_accumulated_mse": best_metric,
                    "resolved_config": config,
                    "rng_state": _rng_state(),
                    "loader_generator_state": loader_generator.get_state(),
                    "window_generator_state": window_generator.get_state(),
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}", flush=True)
    print(f"Training complete; best_validation_accumulated_mse={best_metric:.8e}", flush=True)


if __name__ == "__main__":
    main()
