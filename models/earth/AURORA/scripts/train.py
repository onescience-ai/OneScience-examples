"""Train or resume the OneScience Aurora adapter.

The official Aurora repository does not publish a complete pretraining recipe. This entry point
therefore implements an explicit engineering contract: AdamW, optional bfloat16 autocast,
channel-scale-balanced MSE, periodic validation, and rank-zero checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import h5py
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | Path, *, config_path: Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.resolve().parents[1] / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--static-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("train_from_scratch", "finetune", "resume"), default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-type",
        choices=("official", "training"),
        default=None,
        help="Checkpoint schema used by finetune: official Aurora .ckpt or project training .pt",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--backend", default=None, help="DDP backend; defaults to config")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Require a torchrun DDP environment (also inferred from WORLD_SIZE)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build the loader/model and inspect one batch")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args(argv)


def is_distributed_requested(args: argparse.Namespace) -> bool:
    return args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(args: argparse.Namespace, cfg: Mapping[str, Any]) -> dict[str, Any]:
    distributed = is_distributed_requested(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    backend = args.backend or str(cfg["distributed"].get("backend", "nccl"))
    if distributed and world_size <= 1 and "MASTER_ADDR" not in os.environ:
        raise RuntimeError("Distributed training must be launched with torchrun (MASTER_ADDR is missing)")
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "backend": backend,
        "is_main": rank == 0,
    }


def select_device(args: argparse.Namespace, runtime: Mapping[str, Any]) -> torch.device:
    requested = args.device
    if requested == "auto":
        if torch.cuda.is_available():
            requested = f"cuda:{runtime['local_rank']}"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/DCU device requested but torch.cuda.is_available() is false")
        if device.index is None:
            device = torch.device(device.type, runtime["local_rank"])
    return device


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def configure_logging(output_dir: Path, is_main: bool) -> logging.Logger:
    logger = logging.getLogger("aurora.train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO if is_main else logging.WARNING)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def extract_latest_history_times(time_index: Any, batch_size: int, history_steps: int) -> list[str]:
    """Handle the nested list/tuple form produced by PyTorch's default collate."""
    if isinstance(time_index, (list, tuple)) and len(time_index) > history_steps - 1:
        value = time_index[history_steps - 1]
    else:
        value = time_index
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    if len(values) == 1 and batch_size > 1:
        values *= batch_size
    if len(values) != batch_size:
        raise ValueError(f"Could not extract {batch_size} latest-history timestamps from {time_index!r}")
    return values


def make_loader(
    dataset_dir: Path,
    years: Sequence[int],
    channels: Sequence[str],
    *,
    mode: str,
    input_steps: int,
    output_steps: int,
    normalize: bool,
    batch_size: int,
    num_workers: int,
    distributed: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    from onescience.datapipes.climate import ERA5Dataset

    dataset = ERA5Dataset(
        dataset_dir=str(dataset_dir),
        used_years=list(years),
        used_variables=list(channels),
        mode=mode,
        input_steps=input_steps,
        output_steps=output_steps,
        normalize=normalize,
    )
    sampler = DistributedSampler(dataset, shuffle=mode == "train") if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and mode == "train"),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=distributed and mode == "train",
    )
    return loader, sampler


def build_model(
    cfg: Mapping[str, Any],
    config_path: Path,
    mode: str,
    checkpoint: Path | None,
    static_file: Path,
    checkpoint_type: str,
):
    from model.aurora import build_aurora_model

    project_root = config_path.resolve().parents[1]
    model_cfg = copy.deepcopy(cfg)
    model_cfg["data"]["static_file"] = str(static_file)
    model = build_aurora_model(model_cfg, project_root=project_root, load_pretrained=False)
    if mode in {"train_from_scratch", "finetune"} and checkpoint is not None:
        if checkpoint_type == "official":
            model.load_checkpoint_local(checkpoint, strict=True)
        else:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping) or payload.get("schema_version") != "aurora-training-checkpoint-v1":
                raise ValueError(
                    "--checkpoint-type training requires an aurora-training-checkpoint-v1 checkpoint"
                )
            state = payload.get("model_state")
            if not isinstance(state, Mapping):
                raise ValueError("Training checkpoint does not contain a model_state mapping")
            model.load_state_dict(state, strict=True)
    elif mode == "finetune":
        raise ValueError(
            "No fine-tuning checkpoint was configured. Provide --checkpoint or set "
            "training.finetune.checkpoint."
        )
    return model


def make_autocast(device: torch.device, cfg: Mapping[str, Any]):
    precision = str(cfg["training"].get("precision", "float32"))
    enabled = precision == "bfloat16_autocast" and device.type == "cuda"
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)


def load_channel_scales(
    dataset_dir: Path,
    years: Sequence[int],
    channels: Sequence[str],
    *,
    enabled: bool,
    epsilon: float,
) -> torch.Tensor | None:
    """Load per-channel standard deviations used to balance physical-space errors."""
    if not enabled:
        return None
    yearly_stds = []
    for year in years:
        path = dataset_dir / "data" / f"{year}.h5"
        with h5py.File(path, "r") as handle:
            values = handle["fields"].attrs["variables"]
            names = [value.decode() if isinstance(value, bytes) else str(value) for value in values]
            if names != list(channels):
                raise ValueError(f"{path}: channel order does not match training configuration")
            yearly_stds.append(np.asarray(handle["global_stds"][:], dtype=np.float32).reshape(-1))
    scales = np.sqrt(np.mean(np.square(np.stack(yearly_stds, axis=0)), axis=0))
    return torch.from_numpy(np.maximum(scales, epsilon)).reshape(1, -1, 1, 1)


def mse_assumption(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute balanced MSE; this is an engineering objective, not the unpublished official loss."""
    error = prediction.float() - target.float()
    if channel_scales is not None:
        error = error / channel_scales.to(device=error.device, dtype=error.dtype)
    return torch.mean(error**2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    step: int,
    mode: str,
    config_path: Path,
) -> None:
    module = model.module if isinstance(model, DDP) else model
    payload = {
        "schema_version": "aurora-training-checkpoint-v1",
        "step": step,
        "mode": mode,
        "config": str(config_path),
        "model_state": module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != "aurora-training-checkpoint-v1":
        raise ValueError("--mode resume requires an Aurora training checkpoint produced by this entry point")
    module = model.module if isinstance(model, DDP) else model
    module.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return int(payload["step"])


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    history_steps: int,
    patch_size: int,
    channel_scales: torch.Tensor | None,
) -> float:
    model.eval()
    total = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    with torch.no_grad():
        for invar, outvar, _, _, time_index in loader:
            invar = invar.to(device=device, dtype=torch.float32, non_blocking=True)
            outvar = outvar.to(device=device, dtype=torch.float32, non_blocking=True)
            times = extract_latest_history_times(time_index, invar.shape[0], history_steps)
            with make_autocast(device, {"training": {"precision": "float32"}}):
                prediction = model(invar, times=times)
            target = model.module.crop_target(outvar, patch_size) if isinstance(model, DDP) else model.crop_target(outvar, patch_size)
            total += mse_assumption(prediction, target, channel_scales).detach()
            count += 1
    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return float((total / count.clamp_min(1)).item())


def run_training(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    mode = args.mode or str(cfg["training"].get("mode", "train_from_scratch"))
    runtime = setup_distributed(args, cfg)
    device = select_device(args, runtime)
    seed = int(args.seed if args.seed is not None else cfg["project"]["seed"])
    seed_everything(seed, runtime["rank"])

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_dir = args.data_dir or resolve_path(data_cfg["virtual_dir"], config_path=config_path, project_root=PROJECT_ROOT)
    if args.static_file is not None:
        static_file = args.static_file.resolve()
    elif args.data_dir is not None:
        static_file = (data_dir / "static" / "static_vars.npz").resolve()
    else:
        static_file = resolve_path(data_cfg["static_file"], config_path=config_path, project_root=PROJECT_ROOT)
    configured_log_dir = resolve_path(
        train_cfg["log_dir"], config_path=config_path, project_root=PROJECT_ROOT
    )
    output_dir = (args.output_dir or configured_log_dir.parent).resolve()
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else resolve_path(
            train_cfg["checkpoint_dir"], config_path=config_path, project_root=PROJECT_ROOT
        )
    )
    log_dir = (
        output_dir / "logs"
        if args.output_dir is not None
        else configured_log_dir
    )
    logger = configure_logging(log_dir, runtime["is_main"])
    logger.info("runtime=%s device=%s mode=%s data=%s", runtime, device, mode, data_dir)

    batch_size = int(args.batch_size if args.batch_size is not None else data_cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg["num_workers"])
    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg["max_steps"])
    validation_interval = int(
        args.validation_interval
        if args.validation_interval is not None
        else train_cfg.get("validation_interval", max_steps)
    )
    checkpoint_interval = int(
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else train_cfg.get("checkpoint_interval", max_steps)
    )
    if max_steps < 1 or validation_interval < 1 or checkpoint_interval < 1:
        raise ValueError("max_steps, validation_interval and checkpoint_interval must be positive")
    input_steps = int(data_cfg["input_steps"])
    output_steps = int(data_cfg["output_steps"])
    patch_size = int(model_cfg["patch_size"])
    channels = list(data_cfg["channel_order"])

    train_loader, train_sampler = make_loader(
        data_dir,
        data_cfg["train_years"],
        channels,
        mode="train",
        input_steps=input_steps,
        output_steps=output_steps,
        normalize=bool(data_cfg["normalize_in_onescience"]),
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=runtime["distributed"],
    )
    val_loader, _ = make_loader(
        data_dir,
        data_cfg["val_years"],
        channels,
        mode="val",
        input_steps=input_steps,
        output_steps=output_steps,
        normalize=bool(data_cfg["normalize_in_onescience"]),
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=runtime["distributed"],
    )
    scratch_cfg = train_cfg.get("scratch", {})
    finetune_cfg = train_cfg.get("finetune", {})
    checkpoint = args.checkpoint
    if checkpoint is None and mode == "train_from_scratch" and scratch_cfg.get("checkpoint"):
        checkpoint = Path(str(scratch_cfg["checkpoint"]))
    if checkpoint is None and mode == "finetune" and finetune_cfg.get("checkpoint"):
        checkpoint = Path(str(finetune_cfg["checkpoint"]))
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = (config_path.resolve().parents[1] / checkpoint).resolve()
    checkpoint_type = str(
        args.checkpoint_type
        or (
            finetune_cfg.get("checkpoint_type") if mode == "finetune"
            else scratch_cfg.get("checkpoint_type") if mode == "train_from_scratch"
            else None
        )
        or ("training" if mode == "resume" else "official")
    )
    if mode == "resume":
        if checkpoint_type != "training":
            raise ValueError("--mode resume requires --checkpoint-type training")
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --mode resume")
    model = build_model(cfg, config_path, mode, checkpoint, static_file, checkpoint_type)
    model.to(device)
    if bool(train_cfg.get("gradient_checkpointing", False)) and not args.no_gradient_checkpointing:
        model.configure_activation_checkpointing()
    if runtime["distributed"]:
        device_ids = [device.index] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=False)

    learning_rate = float(
        finetune_cfg.get("learning_rate", train_cfg["learning_rate"])
        if mode == "finetune"
        else train_cfg["learning_rate"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = None
    if str(train_cfg.get("scheduler", "none")).lower() not in {"none", "null", ""}:
        raise ValueError("Only scheduler=none is implemented because the official scheduler is unpublished")
    start_step = 0
    if mode == "resume":
        start_step = load_resume_checkpoint(checkpoint, model, optimizer, scheduler, device)

    loss_cfg = train_cfg.get("loss", {})
    channel_scales = load_channel_scales(
        data_dir,
        data_cfg["train_years"],
        channels,
        enabled=bool(loss_cfg.get("scale_by_channel_std", True)),
        epsilon=float(loss_cfg.get("scale_epsilon", 1.0e-6)),
    )
    logger.info(
        "initialization=%s checkpoint=%s lr=%.3e loss=%s",
        "random" if checkpoint is None else checkpoint_type,
        checkpoint,
        learning_rate,
        loss_cfg.get("name", "channel_std_normalized_mse"),
    )

    if args.dry_run:
        first = next(iter(train_loader))
        invar, outvar, _, _, time_index = first
        times = extract_latest_history_times(time_index, invar.shape[0], input_steps)
        logger.info("dry_run input=%s target=%s times=%s", tuple(invar.shape), tuple(outvar.shape), times)
        if runtime["distributed"]:
            dist.barrier()
            dist.destroy_process_group()
        return 0

    metrics_path = log_dir / "metrics.jsonl"
    step = start_step
    epoch = 0
    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        model.train()
        for invar, outvar, _, _, time_index in train_loader:
            if step >= max_steps:
                break
            invar = invar.to(device=device, dtype=torch.float32, non_blocking=True)
            outvar = outvar.to(device=device, dtype=torch.float32, non_blocking=True)
            times = extract_latest_history_times(time_index, invar.shape[0], input_steps)
            optimizer.zero_grad(set_to_none=True)
            with make_autocast(device, cfg):
                prediction = model(invar, times=times)
                base_model = model.module if isinstance(model, DDP) else model
                target = base_model.crop_target(outvar, patch_size)
                loss = mse_assumption(prediction, target, channel_scales)
                physical_mse = mse_assumption(prediction, target)
            loss.backward()
            clip_norm = float(train_cfg.get("gradient_clip_norm", 0.0))
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            step += 1

            if runtime["is_main"]:
                record = {
                    "step": step,
                    "train_loss": float(loss.detach().item()),
                    "physical_mse": float(physical_mse.detach().item()),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                logger.info(
                    "step=%d loss=%.6e physical_mse=%.6e",
                    step,
                    record["train_loss"],
                    record["physical_mse"],
                )

            if step % validation_interval == 0 or step == max_steps:
                val_loss = evaluate(
                    model,
                    val_loader,
                    device,
                    history_steps=input_steps,
                    patch_size=patch_size,
                    channel_scales=channel_scales,
                )
                if runtime["is_main"]:
                    record = {"step": step, "val_loss": val_loss}
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    logger.info("step=%d val_loss=%.6e", step, val_loss)
                model.train()

            if runtime["is_main"] and (step % checkpoint_interval == 0 or step == max_steps):
                checkpoint_name = "model_finetune.pt" if mode == "finetune" else "model_bak.pt"
                save_checkpoint(
                    checkpoint_dir / checkpoint_name,
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    mode=mode,
                    config_path=config_path,
                )
        if runtime["distributed"]:
            dist.barrier()

    if runtime["is_main"]:
        summary = {
            "schema_version": "aurora-training-summary-v1",
            "status": "completed",
            "mode": mode,
            "steps": step,
            "output_dir": str(output_dir),
            "loss": loss_cfg.get("name", "channel_std_normalized_mse_assumption"),
            "loss_note": train_cfg.get("loss_note", ""),
        }
        with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    if runtime["distributed"]:
        dist.destroy_process_group()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
