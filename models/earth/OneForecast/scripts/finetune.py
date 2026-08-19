"""Multi-step autoregressive fine-tuning for OneForecast."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.era5_adapter import OFFICIAL_VARIABLES, OneForecastERA5Adapter
from model.oneforecast import build_model, read_official_checkpoint
from scripts.train import _LossScaleFunction, _reduce_metrics, _relative_channel_l2, _set_seed, _setup_distributed


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _prepare_batch(batch: tuple, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch[0], batch[1]
    if inputs.ndim != 4 or targets.ndim != 5:
        raise ValueError(f"Expected [B,C,H,W] inputs and [B,S,C,H,W] targets, got {inputs.shape} and {targets.shape}")
    if targets.shape[1] != steps:
        raise ValueError(f"Expected {steps} target steps, got {targets.shape[1]}")
    if inputs.shape[-2] == 121:
        inputs = inputs[..., :120, :]
    if targets.shape[-2] == 121:
        targets = targets[..., :120, :]
    if inputs.shape[-2:] != (120, 240) or targets.shape[-2:] != (120, 240):
        raise ValueError(f"Expected official model grid 120x240, got {inputs.shape} and {targets.shape}")
    return torch.nan_to_num(inputs.float()), torch.nan_to_num(targets.float())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("conf/config.yaml"))
    parser.add_argument("--model-source", choices=("trained", "official"), default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if tuple(config["datapipe"]["variables"]) != OFFICIAL_VARIABLES:
        raise ValueError("datapipe.variables must exactly match the official 69-channel order")

    settings = config["datapipe"]
    finetune = config["finetuning"]
    if args.max_epochs is not None:
        finetune["max_epoch"] = args.max_epochs
    if args.max_batches is not None:
        finetune["max_batches"] = args.max_batches
    steps = int(finetune["steps"])
    if steps < 2:
        raise ValueError("finetuning.steps must be at least 2")
    source = args.model_source or finetune.get("model_source", "trained")
    checkpoint_path = _resolve_path(
        finetune["trained_model_path"] if source == "trained" else finetune["official_checkpoint_path"],
        config_path,
    )
    output_path = _resolve_path(finetune["output_path"], config_path)
    dataset_dir = _resolve_path(settings["dataset_dir"], config_path)

    device, rank, world_size, distributed = _setup_distributed(
        config["runtime"].get("device", "cpu"), config["runtime"].get("distributed_backend", "nccl")
    )
    _set_seed(int(config["runtime"].get("seed", 42)))
    config["model"]["weight_init"] = "scratch"
    model = build_model(config).to(device)
    state, _ = read_official_checkpoint(checkpoint_path)
    model.load_state_dict(state)
    if distributed:
        ddp_devices = {"device_ids": [device.index], "output_device": device.index} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, broadcast_buffers=False, **ddp_devices)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(finetune["learning_rate"]))
    adapter = OneForecastERA5Adapter(
        dataset_dir, settings["train_years"], batch_size=settings["batch_size"],
        input_steps=1, output_steps=steps, normalize=settings["normalize"],
        num_workers=settings["num_workers"], distributed=distributed,
    )
    loader, sampler = adapter.get_dataloader("train")
    max_batches = int(finetune.get("max_batches", -1))

    for epoch in range(int(finetune["max_epoch"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            inputs, targets = _prepare_batch(batch, steps)
            current = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for step in range(steps):
                current = checkpoint(model, current, use_reentrant=False)
                scaled = _LossScaleFunction.apply(current, 1e-5)
                step_loss, _ = _relative_channel_l2(scaled, targets[:, step])
                loss = loss + step_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
            if max_batches >= 0 and batches >= max_batches:
                break
        mean_loss = _reduce_metrics(epoch_loss, batches, device, distributed)
        if rank == 0:
            print({"epoch": epoch + 1, "steps": steps, "loss": mean_loss,
                   "batches_per_rank": batches, "world_size": world_size})

    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state = model.module.state_dict() if distributed else model.state_dict()
        torch.save({"model_state": state, "epoch": int(finetune["max_epoch"]),
                    "finetune_steps": steps, "world_size": world_size}, output_path)
        print({"checkpoint": str(output_path)})
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
