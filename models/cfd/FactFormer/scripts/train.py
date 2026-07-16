from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LOCAL_WORKSPACE = PROJECT_ROOT.parents[1]
if (LOCAL_WORKSPACE / "onescience" / "src" / "onescience" / "datapipes").is_dir():
    sys.path.insert(0, str(LOCAL_WORKSPACE))

from common import (  # noqa: E402
    load_config,
    project_path,
    relative_l2,
    resolve_device,
    rollout,
    seed_everything,
    to_attr_dict,
    to_plain_dict,
)
from model.factformer import FactFormer2D  # noqa: E402
from onescience.datapipes.cfd import KolmogorovFlow2DDatapipe  # noqa: E402
from onescience.distributed.manager import DistributedManager  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def prepare_config(config: dict[str, Any]) -> None:
    datapipe = config["datapipe"]
    data = datapipe["data"]
    model = config["model"]
    datapipe["source"]["data_dir"] = str(
        project_path(datapipe["source"]["data_dir"], PROJECT_ROOT).resolve()
    )
    stats_file = data.get("stats_file")
    if stats_file:
        data["stats_file"] = str(project_path(stats_file, PROJECT_ROOT).resolve())
    model["in_dim"] = int(data["t_in"]) * int(data["out_dim"])
    model["out_dim"] = int(data["out_dim"])


def validate_config(config: dict[str, Any]) -> None:
    data = config["datapipe"]["data"]
    loader = config["datapipe"]["dataloader"]
    model = config["model"]
    training = config["training"]
    positive_values = {
        "train_num": data["train_num"],
        "test_num": data["test_num"],
        "resolution": data["resolution"],
        "interval": data["interval"],
        "t_in": data["t_in"],
        "t_out": data["t_out"],
        "batch_size": loader["batch_size"],
        "hidden_dim": model["hidden_dim"],
        "depth": model["depth"],
        "heads": model["heads"],
        "mlp_ratio": model["mlp_ratio"],
        "max_latent_steps": model["max_latent_steps"],
        "epochs": training["epochs"],
        "eval_interval": training["eval_interval"],
        "train_latent_steps": training["train_latent_steps"],
    }
    for name, value in positive_values.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    if int(loader["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    if int(model["hidden_dim"]) % int(model["heads"]):
        raise ValueError("hidden_dim must be divisible by heads")
    if int(training["train_latent_steps"]) > int(model["max_latent_steps"]):
        raise ValueError("train_latent_steps cannot exceed model.max_latent_steps")
    if int(training["train_latent_steps"]) > int(data["t_out"]):
        raise ValueError("train_latent_steps cannot exceed datapipe.data.t_out")
    for name in ("max_train_batches", "max_eval_batches"):
        value = training.get(name)
        if value is not None and int(value) < 1:
            raise ValueError(f"{name} must be positive when set")


def build_model(
    model_config: dict[str, Any], spatial_shape: tuple[int, int]
) -> FactFormer2D:
    return FactFormer2D(
        in_dim=int(model_config["in_dim"]),
        out_dim=int(model_config["out_dim"]),
        spatial_shape=spatial_shape,
        hidden_dim=int(model_config["hidden_dim"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        mlp_ratio=int(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
        activation=str(model_config["activation"]),
        include_pos=bool(model_config["include_pos"]),
        space_dim=int(model_config["space_dim"]),
        latent_multiplier=float(model_config["latent_multiplier"]),
        max_latent_steps=int(model_config["max_latent_steps"]),
    )


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    datapipe: KolmogorovFlow2DDatapipe,
    t_out: int,
    out_dim: int,
    max_latent_steps: int,
    max_batches: int | None,
    distributed: bool,
    rank: int,
) -> tuple[float, float]:
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    with torch.no_grad():
        iterator = tqdm(loader, desc="Evaluating", disable=rank != 0)
        for batch_index, batch in enumerate(iterator):
            if max_batches is not None and batch_index >= max_batches:
                break
            pos = batch["pos"].to(device)
            state = batch["x"].to(device)
            target = batch["y"].to(device)
            prediction = rollout(
                model, pos, state, t_out, out_dim, max_latent_steps
            )
            prediction = datapipe.decode_solution(prediction)
            target = datapipe.decode_solution(target)
            totals[0] += F.mse_loss(prediction, target, reduction="sum")
            totals[1] += target.numel()
            totals[2] += relative_l2(prediction, target).sum()
            totals[3] += target.shape[0]
    if distributed:
        torch.distributed.all_reduce(totals)
    return (
        (totals[0] / totals[1].clamp_min(1)).item(),
        (totals[2] / totals[3].clamp_min(1)).item(),
    )


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    prepare_config(config)
    validate_config(config)
    common = config["common"]
    datapipe_config = config["datapipe"]
    model_config = config["model"]
    training = config["training"]
    seed_everything(int(common["seed"]))

    DistributedManager.initialize()
    dist = DistributedManager()
    distributed = dist.world_size > 1
    device = dist.device if str(common["device"]) == "auto" else resolve_device(
        str(common["device"])
    )
    weight_dir = project_path(training["weight_dir"], PROJECT_ROOT).resolve()
    checkpoint_path = weight_dir / str(training["checkpoint_name"])

    if dist.rank == 0:
        print(f"Config: {config_path}")
        print(
            "Data: "
            f"{Path(datapipe_config['source']['data_dir']) / datapipe_config['source']['file_name']}"
        )
        print(f"Device: {device}")
        print(
            f"Samples: train={datapipe_config['data']['train_num']} "
            f"test={datapipe_config['data']['test_num']} "
            f"t_in={datapipe_config['data']['t_in']} "
            f"t_out={datapipe_config['data']['t_out']}"
        )

    started = time.time()
    try:
        datapipe = KolmogorovFlow2DDatapipe(
            to_attr_dict(datapipe_config), distributed=distributed
        )
        train_loader, train_sampler = datapipe.train_dataloader()
        test_loader, _ = datapipe.test_dataloader()
        spatial_shape = tuple(datapipe.spatial_shape)
        model = build_model(model_config, spatial_shape).to(device)
        if distributed:
            device_ids = [dist.local_rank] if device.type == "cuda" else None
            model = DDP(model, device_ids=device_ids)
        if dist.rank == 0:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            print(f"Spatial shape: {spatial_shape}")
            print(f"Parameters: {parameter_count:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training["step_size"]),
            gamma=float(training["gamma"]),
        )
        data = datapipe_config["data"]
        t_out = int(data["t_out"])
        out_dim = int(data["out_dim"])
        max_latent_steps = int(model_config["max_latent_steps"])
        train_latent_steps = int(training["train_latent_steps"])
        max_train_batches = training.get("max_train_batches")
        max_train_batches = None if max_train_batches is None else int(max_train_batches)
        max_eval_batches = training.get("max_eval_batches")
        max_eval_batches = None if max_eval_batches is None else int(max_eval_batches)
        best_relative_l2 = float("inf")
        stale_evaluations = 0

        for epoch in range(1, int(training["epochs"]) + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=dist.rank != 0)
            for batch_index, batch in enumerate(iterator):
                if max_train_batches is not None and batch_index >= max_train_batches:
                    break
                pos = batch["pos"].to(device)
                state = batch["x"].to(device)
                target = batch["y"].to(device)[..., : train_latent_steps * out_dim]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(pos, state, latent_steps=train_latent_steps)
                loss = F.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                if training.get("max_grad_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(training["max_grad_norm"])
                    )
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
                if dist.rank == 0:
                    iterator.set_postfix(loss=f"{loss.item():.3e}")
            scheduler.step()
            if batch_count == 0:
                raise RuntimeError("No training batches were processed")

            should_evaluate = (
                epoch % int(training["eval_interval"]) == 0
                or epoch == int(training["epochs"])
            )
            if not should_evaluate:
                continue
            validation_mse, validation_relative_l2 = evaluate(
                model,
                test_loader,
                device,
                datapipe,
                t_out,
                out_dim,
                max_latent_steps,
                max_eval_batches,
                distributed,
                dist.rank,
            )
            if dist.rank == 0:
                print(
                    f"epoch={epoch:4d} train_mse={epoch_loss / batch_count:.6e} "
                    f"val_mse={validation_mse:.6e} "
                    f"val_relative_l2={validation_relative_l2:.6e}"
                )
                if validation_relative_l2 < best_relative_l2:
                    best_relative_l2 = validation_relative_l2
                    stale_evaluations = 0
                    weight_dir.mkdir(parents=True, exist_ok=True)
                    model_to_save = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model_to_save.state_dict(),
                            "model_config": to_plain_dict(model_config),
                            "datapipe_config": to_plain_dict(datapipe_config),
                            "training_config": to_plain_dict(training),
                            "spatial_shape": spatial_shape,
                            "normalizer": datapipe.get_normalizer_state(),
                            "best_relative_l2": best_relative_l2,
                        },
                        checkpoint_path,
                    )
                    print(f"Saved checkpoint: {checkpoint_path}")
                else:
                    stale_evaluations += 1

            stop = torch.tensor(
                [int(stale_evaluations >= int(training["patience"]))], device=device
            )
            if distributed:
                torch.distributed.broadcast(stop, src=0)
            if stop.item():
                if dist.rank == 0:
                    print("Early stopping triggered")
                break

        if dist.rank == 0:
            print(f"Elapsed: {time.time() - started:.1f}s")
    finally:
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
