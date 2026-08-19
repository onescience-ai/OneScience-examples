"""OneForecast training entry point with integrated data checking."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.era5_adapter import OFFICIAL_VARIABLES, OneForecastERA5Adapter
from model.oneforecast import build_model, check_checkpoint_compatibility


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["datapipe"]["dataset_dir"] = str(_resolve_path(config["datapipe"]["dataset_dir"], path))
    config["model"]["official_checkpoint_path"] = str(
        _resolve_path(config["model"]["official_checkpoint_path"], path)
    )
    config["model"]["checkpoint_path"] = config["model"]["official_checkpoint_path"]
    config["training"]["checkpoint_dir"] = str(_resolve_path(config["training"]["checkpoint_dir"], path))
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(name: str) -> torch.device:
    """Map the logical DCU name to the backend exposed by this PyTorch build."""
    requested = str(name).lower()
    if requested == "dcu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        privateuse = torch._C._get_privateuse1_backend_name()
        if privateuse != "privateuseone":
            return torch.device(privateuse)
        raise RuntimeError("runtime.device=dcu, but this PyTorch build exposes no usable accelerator")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda, but torch.cuda.is_available() is False")
    return device


def _setup_distributed(device_name: str, backend: str = "nccl") -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        device = _resolve_device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        return device, dist.get_rank(), world_size, True
    return _resolve_device(device_name), 0, 1, False


def _reduce_metrics(total: float, count: int, device: torch.device, distributed: bool) -> float:
    metrics = torch.tensor([total, count], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return float(metrics[0] / metrics[1].clamp_min(1))


def _loader_batch(batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch[0], batch[1]
    if inputs.ndim == 5 or targets.ndim == 5:
        raise ValueError("OneForecast currently supports input_steps=1 and output_steps=1 only")
    if inputs.ndim != 4 or targets.ndim != 4:
        raise ValueError(f"Expected batched fields with four dimensions, got {inputs.shape} and {targets.shape}")
    if inputs.shape[-2] == 121:
        inputs = inputs[..., :120, :]
    if targets.shape[-2] == 121:
        targets = targets[..., :120, :]
    if inputs.shape[-2:] != (120, 240) or targets.shape[-2:] != (120, 240):
        raise ValueError(f"Expected official model grid 120x240, got {inputs.shape} and {targets.shape}")
    return torch.nan_to_num(inputs.float()), torch.nan_to_num(targets.float())


class _LossScaleFunction(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, eps: float) -> torch.Tensor:
        ctx.eps = eps
        return values

    @staticmethod
    def backward(ctx, gradients: torch.Tensor) -> tuple[torch.Tensor, None]:
        channels = gradients.shape[1]
        weights = 1.0 / gradients.norm(p=2, dim=(-1, -2), keepdim=True).clamp_min(ctx.eps)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(ctx.eps)
        return channels * weights * gradients, None


def _relative_channel_l2(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    difference = (prediction - target).flatten(2).norm(p=2, dim=2)
    target_norm = target.flatten(2).norm(p=2, dim=2).clamp_min(1e-10)
    channel_loss = (difference / target_norm).mean(dim=0)
    return channel_loss.mean(), channel_loss


def check_data(config: dict) -> dict:
    settings = config["datapipe"]
    adapter = OneForecastERA5Adapter(
        settings["dataset_dir"], settings["train_years"],
        batch_size=settings["batch_size"], input_steps=settings["input_steps"],
        output_steps=settings["output_steps"], normalize=settings["normalize"],
        num_workers=settings["num_workers"],
    )
    report = adapter.inspect()
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("conf/config.yaml"))
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--check-checkpoint", action="store_true")
    parser.add_argument("--check-distributed", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--distributed-backend", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--weight-init", choices=("scratch", "official"), default=None)
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.distributed_backend is not None:
        config["runtime"]["distributed_backend"] = args.distributed_backend
    if args.max_epochs is not None:
        config["training"]["max_epoch"] = args.max_epochs
    if args.max_batches is not None:
        config["training"]["max_batches"] = args.max_batches
    if tuple(config["datapipe"]["variables"]) != OFFICIAL_VARIABLES:
        raise ValueError("datapipe.variables must exactly match the official 69-channel order")
    if args.weight_init is not None:
        config["model"]["weight_init"] = args.weight_init
    if config["model"].get("weight_init") == "official":
        config["model"]["checkpoint_path"] = config["model"]["official_checkpoint_path"]
    if args.check_data:
        check_data(config)
        return
    if args.check_model:
        configured_init = config["model"].get("weight_init", "scratch")
        config["model"]["weight_init"] = "scratch"
        with __import__("torch").device("meta"):
            model = build_model(config, build_graph=False)
        print({"model": type(model).__name__, "parameters": sum(p.numel() for p in model.parameters()),
               "configured_weight_init": configured_init})
        return
    if args.check_checkpoint:
        with __import__("torch").device("meta"):
            model = build_model(config, build_graph=False)
        report = check_checkpoint_compatibility(
            model, config["model"]["official_checkpoint_path"]
        )
        print(report)
        if not report.compatible:
            raise SystemExit(1)
        return
    if args.check_distributed:
        device, rank, world_size, distributed = _setup_distributed(
            config["runtime"].get("device", "cpu"), config["runtime"].get("distributed_backend", "nccl")
        )
        settings = config["datapipe"]
        adapter = OneForecastERA5Adapter(
            settings["dataset_dir"], settings["train_years"], batch_size=settings["batch_size"],
            input_steps=settings["input_steps"], output_steps=settings["output_steps"],
            normalize=settings["normalize"], num_workers=0, distributed=distributed,
        )
        loader, sampler = adapter.get_dataloader("train")
        sample_indices = list(iter(sampler)) if sampler is not None else list(range(len(loader.dataset)))
        print({"rank": rank, "world_size": world_size, "distributed": distributed,
               "backend": dist.get_backend() if distributed else None, "device": str(device),
               "sampler": type(sampler).__name__ if sampler is not None else None,
               "sample_indices": sample_indices})
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    settings = config["datapipe"]
    if settings["input_steps"] != 1 or settings["output_steps"] != 1:
        raise SystemExit("OneForecast training currently requires input_steps=1 and output_steps=1")
    device, rank, world_size, distributed = _setup_distributed(
        config["runtime"].get("device", "cpu"), config["runtime"].get("distributed_backend", "nccl")
    )
    _set_seed(int(config["runtime"].get("seed", 42)))
    model = build_model(config).to(device)
    if distributed:
        ddp_devices = {"device_ids": [device.index], "output_device": device.index} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, broadcast_buffers=False, **ddp_devices)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config["training"]["max_epoch"])),
    )
    train_adapter = OneForecastERA5Adapter(
        _resolve_path(settings["dataset_dir"], args.config), settings["train_years"],
        batch_size=settings["batch_size"], input_steps=1, output_steps=1,
        normalize=settings["normalize"], num_workers=settings["num_workers"], distributed=distributed,
    )
    valid_adapter = OneForecastERA5Adapter(
        _resolve_path(settings["dataset_dir"], args.config), settings["valid_years"],
        batch_size=settings["batch_size"], input_steps=1, output_steps=1,
        normalize=settings["normalize"], num_workers=settings["num_workers"], distributed=distributed,
    )
    train_loader, train_sampler = train_adapter.get_dataloader("train")
    valid_loader, valid_sampler = valid_adapter.get_dataloader("val")
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    max_batches = int(config["training"].get("max_batches", -1))
    for epoch in range(int(config["training"]["start_epoch"]), int(config["training"]["max_epoch"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if valid_sampler is not None:
            valid_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            inputs, targets = _loader_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = _LossScaleFunction.apply(model(inputs.to(device)), 1e-5)
            loss, _ = _relative_channel_l2(prediction, targets.to(device))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            train_batches += 1
            if max_batches >= 0 and train_batches >= max_batches:
                break
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            valid_batches = 0
            for batch in valid_loader:
                inputs, targets = _loader_batch(batch)
                prediction = model(inputs.to(device))
                valid_loss += float(F.mse_loss(prediction, targets.to(device)))
                valid_batches += 1
                if max_batches >= 0 and valid_batches >= max_batches:
                    break
        train_mean = _reduce_metrics(train_loss, train_batches, device, distributed)
        valid_mean = _reduce_metrics(valid_loss, valid_batches, device, distributed)
        if rank == 0:
            print({"epoch": epoch + 1, "train_loss": train_mean, "valid_loss": valid_mean,
                   "world_size": world_size})
        if rank == 0 and (epoch + 1) % int(config["training"].get("save_every_epoch", 1)) == 0:
            model_name = config["training"].get("model_name", "model_bak")
            state = model.module.state_dict() if distributed else model.state_dict()
            torch.save({"model_state": state, "epoch": epoch + 1, "world_size": world_size},
                       checkpoint_dir / f"{model_name}.tar")
        scheduler.step()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
