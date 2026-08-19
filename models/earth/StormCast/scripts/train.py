from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "model"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, MODEL_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from stormer import build_diffusion_model, build_regression_model
from data_loader import StormCastDataset


@dataclass
class DistributedContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def regression_loss(
    model: nn.Module,
    condition: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(model(condition), target)


def edm_residual_loss(
    model: nn.Module,
    residual: torch.Tensor,
    condition: torch.Tensor,
    sigma_data: float = 0.5,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    sigma = torch.exp(
        torch.randn(residual.shape[0], device=residual.device) * p_std + p_mean
    )
    noise = torch.randn_like(residual) * sigma[:, None, None, None]
    denoised = model(residual + noise, sigma, condition=condition)
    weight = (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()
    return (weight[:, None, None, None] * (denoised - residual).square()).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two-stage StormCast model")
    parser.add_argument("--config", type=Path, default=Path("conf/config.yaml"))
    parser.add_argument("--stage", choices=("regression", "diffusion"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--regression-weights", type=Path)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _resolve_config_paths(config, config_path.parent.parent)

    training = config["training"]
    stage = args.stage or training["stage"]
    resume = args.resume or _optional_path(training.get("resume_checkpoint"))
    initial_weights = args.initial_weights or _optional_path(training.get("initial_weights"))
    regression_weights = args.regression_weights or _optional_path(
        training.get("regression_weights")
    )
    max_steps = args.max_steps if args.max_steps is not None else training["max_steps"]
    if resume is None and initial_weights is None and not training["from_scratch"]:
        checkpoint_key = f"{stage}_checkpoint"
        initial_weights = _optional_path(config["model"].get(checkpoint_key))
        if initial_weights is None:
            raise ValueError(
                f"training.from_scratch is false but model.{checkpoint_key} is not set"
            )

    context = initialize_distributed()
    _seed_everything(config["project"]["seed"], context.rank)
    try:
        train(
            config=config,
            stage=stage,
            context=context,
            resume=resume,
            initial_weights=initial_weights,
            regression_weights=regression_weights,
            max_steps=max_steps,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def train(
    config: dict[str, Any],
    stage: str,
    context: DistributedContext,
    resume: Path | None,
    initial_weights: Path | None,
    regression_weights: Path | None,
    max_steps: int | None,
) -> None:
    if stage not in ("regression", "diffusion"):
        raise ValueError("training.stage must be 'regression' or 'diffusion'")
    data_config = config["data"]
    loader_config = config["dataloader"]
    training_config = config["training"]
    if list(data_config["image_size"]) != list(config["model"]["image_size"]):
        raise ValueError("Data and model image sizes must match")
    if list(data_config["era5_image_size"]) != [721, 1440]:
        raise ValueError("ERA5 grid must be 721 x 1440")

    dataset = StormCastDataset(
        data_root=data_config["root_dir"],
        years=data_config["train_years"],
        era5_variables=data_config["era5_variables"],
        state_variables=data_config["state_variables"],
        invariant_variables=data_config["invariant_variables"],
        image_size=data_config["image_size"],
        input_steps=data_config["input_steps"],
        output_steps=data_config["output_steps"],
        normalize=data_config["normalize"],
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
        )
        if context.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=loader_config["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        drop_last=False,
    )

    regression, model = _build_stage_models(
        config, stage, regression_weights, context.device
    )
    if initial_weights is not None and resume is None:
        _load_initial_weights(model, initial_weights, stage)
    model.to(context.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=tuple(training_config["betas"]),
        weight_decay=training_config["weight_decay"],
    )

    start_epoch = 0
    start_batch = 0
    global_step = 0
    if resume is not None:
        start_epoch, start_batch, global_step = load_training_checkpoint(
            resume, model, optimizer, stage, context.device
        )
    if context.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )

    checkpoint_dir = Path(training_config["checkpoint_dir"]) / stage
    if context.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"stage={stage} world_size={context.world_size} "
            f"parameters={sum(parameter.numel() for parameter in model.parameters())}"
        )

    stop = False
    if max_steps is not None and global_step >= max_steps:
        stop = True
    for epoch in range(start_epoch, training_config["epochs"]):
        if stop:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            loss = _training_step(
                stage,
                model,
                regression,
                batch,
                context.device,
                training_config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_norm = training_config.get("gradient_clip_norm")
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            global_step += 1

            reduced_loss = _mean_across_ranks(loss.detach(), context.world_size)
            if context.is_main and global_step % training_config["log_interval"] == 0:
                print(
                    f"stage={stage} epoch={epoch + 1} step={global_step} "
                    f"loss={reduced_loss.item():.8f}"
                )
            if (
                context.is_main
                and global_step % training_config["checkpoint_interval"] == 0
            ):
                save_training_checkpoint(
                    checkpoint_dir / "model_bak.pt",
                    model,
                    optimizer,
                    stage,
                    epoch,
                    batch_index + 1,
                    global_step,
                    config,
                )
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
        if stop:
            break

    if context.is_main:
        checkpoint = checkpoint_dir / "model_bak.pt"
        save_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            stage,
            epoch if "epoch" in locals() else start_epoch,
            batch_index + 1 if "batch_index" in locals() else start_batch,
            global_step,
            config,
        )
        print(f"checkpoint={checkpoint} steps={global_step}")


def _build_stage_models(
    config: dict[str, Any],
    stage: str,
    regression_weights: Path | None,
    device: torch.device,
) -> tuple[nn.Module | None, nn.Module]:
    data_config = config["data"]
    model_config = config["model"]
    common = {
        "image_size": model_config["image_size"],
        "state_channels": len(data_config["state_variables"]),
        "invariant_channels": len(data_config["invariant_variables"]),
        "model_channels": model_config["model_channels"],
        "channel_mult": model_config["channel_mult"],
        "num_blocks": model_config["num_blocks"],
        "attn_resolutions": model_config["attention_resolutions"],
    }
    if stage == "regression":
        model = build_regression_model(
            **common,
            background_channels=len(data_config["era5_variables"]),
        )
        return None, model

    if regression_weights is None:
        raise ValueError("Diffusion training requires --regression-weights")
    regression = _load_model_weights(
        build_regression_model(
            **common,
            background_channels=len(data_config["era5_variables"]),
        ),
        regression_weights,
        "regression",
    ).to(device)
    regression.eval()
    regression.requires_grad_(False)
    return regression, build_diffusion_model(**common)


def _training_step(
    stage: str,
    model: nn.Module,
    regression: nn.Module | None,
    batch: dict[str, Any],
    device: torch.device,
    training_config: dict[str, Any],
) -> torch.Tensor:
    background = batch["background"].to(device, dtype=torch.float32)
    state, target = (
        tensor.to(device, dtype=torch.float32) for tensor in batch["state"]
    )
    invariant = batch["invariant"].to(device, dtype=torch.float32)
    if invariant.ndim == 3:
        invariant = invariant.unsqueeze(0)
    if invariant.shape[0] == 1 and state.shape[0] > 1:
        invariant = invariant.expand(state.shape[0], -1, -1, -1)

    if stage == "regression":
        condition = torch.cat((state, background, invariant), dim=1)
        return regression_loss(model, condition, target)

    if regression is None:
        raise RuntimeError("Regression model is required for diffusion training")
    with torch.no_grad():
        regression_condition = torch.cat((state, background, invariant), dim=1)
        regression_prediction = regression(regression_condition)
        residual = target - regression_prediction
        condition = torch.cat((state, regression_prediction, invariant), dim=1)
    return edm_residual_loss(
        model,
        residual,
        condition,
        sigma_data=training_config["sigma_data"],
        p_mean=training_config["P_mean"],
        p_std=training_config["P_std"],
    )


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stage: str,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> None:
    model = model.module if isinstance(model, DistributedDataParallel) else model
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": stage,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stage: str,
    device: torch.device,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["stage"] != stage:
        raise ValueError(
            f"Checkpoint stage is {checkpoint['stage']}, requested stage is {stage}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["epoch"]),
        int(checkpoint.get("batch_in_epoch", 0)),
        int(checkpoint["global_step"]),
    )


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("StormCast training requires a CUDA/HIP device")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        rank = 0
    return DistributedContext(
        device=torch.device("cuda", local_rank),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def _load_initial_weights(model: nn.Module, path: Path, stage: str) -> None:
    loaded = _load_model_weights(model, path, stage)
    model.load_state_dict(loaded.state_dict(), strict=True)


def _load_model_weights(model: nn.Module, path: Path, stage: str) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return model


def _mean_across_ranks(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def _resolve_config_paths(config: dict[str, Any], project_root: Path) -> None:
    for section, key in (
        ("data", "root_dir"),
        ("training", "checkpoint_dir"),
    ):
        path = Path(config[section][key])
        if not path.is_absolute():
            config[section][key] = str((project_root / path).resolve())
    for key in ("regression_weights", "diffusion_weights"):
        value = config["model"].get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                config["model"][key] = str((project_root / path).resolve())
    for key in ("initial_weights", "resume_checkpoint", "regression_weights"):
        value = config["training"].get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                config["training"][key] = str((project_root / path).resolve())


def _optional_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


def _seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    main()
