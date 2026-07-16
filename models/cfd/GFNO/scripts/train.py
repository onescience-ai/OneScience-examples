from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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
from model.gfno import GFNO  # noqa: E402
from onescience.datapipes.cfd import NavierStokesDatapipe  # noqa: E402
from onescience.distributed.manager import DistributedManager  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def prepare_config(config: dict[str, Any]) -> None:
    datapipe = config["datapipe"]
    data = datapipe["data"]
    model = config["model"]
    datapipe["source"]["data_dir"] = str(
        project_path(datapipe["source"]["data_dir"], PROJECT_ROOT).resolve()
    )
    model["in_dim"] = int(data["t_in"]) * int(data["out_dim"])
    model["out_dim"] = int(data["out_dim"])


def validate_config(config: dict[str, Any]) -> None:
    data = config["datapipe"]["data"]
    loader = config["datapipe"]["dataloader"]
    model = config["model"]
    training = config["training"]
    positive_values = {
        "ntrain": data["ntrain"],
        "ntest": data["ntest"],
        "t_in": data["t_in"],
        "t_out": data["t_out"],
        "downsamplex": data["downsamplex"],
        "downsampley": data["downsampley"],
        "batch_size": loader["batch_size"],
        "hidden_dim": model["hidden_dim"],
        "modes": model["modes"],
        "num_layers": model["num_layers"],
        "epochs": training["epochs"],
        "eval_interval": training["eval_interval"],
    }
    for name, value in positive_values.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    if int(loader["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    for name in ("max_train_batches", "max_eval_batches"):
        value = training.get(name)
        if value is not None and int(value) < 1:
            raise ValueError(f"{name} must be positive when set")


def build_model(model_config: dict[str, Any], spatial_shape: tuple[int, int]) -> GFNO:
    return GFNO(
        in_dim=int(model_config["in_dim"]),
        out_dim=int(model_config["out_dim"]),
        spatial_shape=spatial_shape,
        hidden_dim=int(model_config["hidden_dim"]),
        modes=int(model_config["modes"]),
        num_layers=int(model_config["num_layers"]),
        space_dim=int(model_config["space_dim"]),
        include_pos=bool(model_config["include_pos"]),
        activation=str(model_config["activation"]),
        reflection=bool(model_config["reflection"]),
        pad_to_multiple=int(model_config["pad_to_multiple"]),
    )


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    t_out: int,
    out_dim: int,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    squared_error = 0.0
    element_count = 0
    relative_error = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if max_batches is not None and batch_index >= max_batches:
                break
            pos = batch["pos"].to(device)
            state = batch["x"].to(device)
            target = batch["y"].to(device)
            prediction = rollout(model, pos, state, t_out, out_dim)
            squared_error += F.mse_loss(prediction, target, reduction="sum").item()
            element_count += target.numel()
            relative_error += relative_l2(prediction, target).sum().item()
            sample_count += target.shape[0]
    return (
        squared_error / max(element_count, 1),
        relative_error / max(sample_count, 1),
    )


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    prepare_config(config)
    validate_config(config)

    common = config["common"]
    data_config = config["datapipe"]
    model_config = config["model"]
    training = config["training"]
    device = resolve_device(str(common["device"]))
    seed_everything(int(common["seed"]))
    weight_dir = project_path(training["weight_dir"], PROJECT_ROOT).resolve()
    checkpoint_path = weight_dir / str(training["checkpoint_name"])

    print(f"Config: {config_path}")
    print(f"Data: {Path(data_config['source']['data_dir']) / data_config['source']['file_name']}")
    print(f"Device: {device}")
    print(
        f"Samples: train={data_config['data']['ntrain']} test={data_config['data']['ntest']} "
        f"t_in={data_config['data']['t_in']} t_out={data_config['data']['t_out']}"
    )

    DistributedManager.initialize()
    started = time.time()
    try:
        datapipe = NavierStokesDatapipe(to_attr_dict(data_config), distributed=False)
        train_loader, _ = datapipe.train_dataloader()
        test_loader, _ = datapipe.test_dataloader()
        model = build_model(model_config, datapipe.spatial_shape).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"Spatial shape: {datapipe.spatial_shape}")
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
        t_out = int(data_config["data"]["t_out"])
        out_dim = int(data_config["data"]["out_dim"])
        max_train_batches = training.get("max_train_batches")
        max_train_batches = None if max_train_batches is None else int(max_train_batches)
        max_eval_batches = training.get("max_eval_batches")
        max_eval_batches = None if max_eval_batches is None else int(max_eval_batches)
        best_relative_l2 = float("inf")
        stale_evaluations = 0

        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch_index, batch in enumerate(iterator):
                if max_train_batches is not None and batch_index >= max_train_batches:
                    break
                pos = batch["pos"].to(device)
                state = batch["x"].to(device)
                target = batch["y"].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = rollout(
                    model,
                    pos,
                    state,
                    t_out,
                    out_dim,
                    teacher_forcing=bool(training["teacher_forcing"]),
                    target=target,
                )
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
                t_out,
                out_dim,
                max_eval_batches,
            )
            print(
                f"epoch={epoch:4d} train_mse={epoch_loss / batch_count:.6e} "
                f"val_mse={validation_mse:.6e} val_relative_l2={validation_relative_l2:.6e}"
            )
            if validation_relative_l2 < best_relative_l2:
                best_relative_l2 = validation_relative_l2
                stale_evaluations = 0
                weight_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "model_config": to_plain_dict(model_config),
                        "datapipe_config": to_plain_dict(data_config),
                        "training_config": to_plain_dict(training),
                        "spatial_shape": tuple(datapipe.spatial_shape),
                        "normalizer": datapipe.get_normalizer_state(),
                        "best_relative_l2": best_relative_l2,
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint: {checkpoint_path}")
            else:
                stale_evaluations += 1
            if stale_evaluations >= int(training["patience"]):
                print("Early stopping triggered")
                break

        print(f"Elapsed: {time.time() - started:.1f}s")
    finally:
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
