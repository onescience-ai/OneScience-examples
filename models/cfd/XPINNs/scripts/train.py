from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import (  # noqa: E402
    build_evaluation_points,
    build_training_batch,
    exact_subdomain_values,
    load_mat_data,
)
from model.xpinn import XPINNPoisson2D  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "root" not in config:
        raise ValueError(f"config must contain a 'root' mapping: {path}")
    return config["root"]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but torch.cuda.is_available() is false")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    boundary_loss = torch.mean(
        (batch["ub"] - outputs["boundary_prediction"]).square()
    )
    pde_loss = sum(
        torch.mean(outputs[f"residual{domain}"].square()) for domain in (1, 2, 3)
    )
    interface_residual_loss = torch.mean(
        outputs["interface1_residual"].square()
    ) + torch.mean(outputs["interface2_residual"].square())
    interface_value_loss = (
        torch.mean(
            (outputs["interface1_domain1"] - outputs["interface1_average"]).square()
        )
        + torch.mean(
            (outputs["interface1_domain2"] - outputs["interface1_average"]).square()
        )
        + torch.mean(
            (outputs["interface2_domain1"] - outputs["interface2_average"]).square()
        )
        + torch.mean(
            (outputs["interface2_domain3"] - outputs["interface2_average"]).square()
        )
    )
    total = (
        float(weights["boundary"]) * boundary_loss
        + float(weights["pde"]) * pde_loss
        + float(weights["interface_residual"]) * interface_residual_loss
        + float(weights["interface_value"]) * interface_value_loss
    )
    return total, {
        "boundary": boundary_loss.item(),
        "pde": pde_loss.item(),
        "interface_residual": interface_residual_loss.item(),
        "interface_value": interface_value_loss.item(),
    }


def clear_coordinate_grads(batch: dict[str, torch.Tensor]) -> None:
    for value in batch.values():
        if value.requires_grad and value.grad is not None:
            value.grad = None


def evaluate(
    model: XPINNPoisson2D,
    points: dict[str, torch.Tensor],
    references: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, float, float, float]:
    with torch.no_grad():
        predictions = model.predict(points["xy1"], points["xy2"], points["xy3"])
        errors = tuple(
            (
                torch.linalg.vector_norm(prediction - reference)
                / torch.linalg.vector_norm(reference)
            ).item()
            for prediction, reference in zip(predictions, references, strict=True)
        )
        combined_prediction = torch.cat(predictions)
        combined_reference = torch.cat(references)
        combined_error = (
            torch.linalg.vector_norm(combined_prediction - combined_reference)
            / torch.linalg.vector_norm(combined_reference)
        ).item()
    return errors[0], errors[1], errors[2], combined_error


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    common = config["common"]
    device = resolve_device(str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    seed = int(common["seed"])
    data_path = project_path(config["data"]["mat_file"])
    weight_dir = project_path(common["weight_dir"])
    checkpoint_path = weight_dir / config["training"]["checkpoint_name"]
    steps = int(config["training"]["steps"])
    if steps <= 0:
        raise ValueError("training steps must be positive")
    counts = {key: int(value) for key, value in config["data"]["samples"].items()}
    seed_everything(seed)

    print(f"Config: {config_path}")
    print(f"Data: {data_path}")
    print(f"Device: {device}")
    print(f"Samples: {counts}")
    data = load_mat_data(data_path)
    batch = build_training_batch(data, counts, seed, device, dtype)
    evaluation_points = build_evaluation_points(data, device, dtype)
    references = exact_subdomain_values(data, device, dtype)

    model = XPINNPoisson2D(config["model"], dtype=dtype).to(
        device=device, dtype=dtype
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["lr"])
    )
    log_interval = int(config["training"]["log_interval"])
    started = time.time()
    for step in range(1, steps + 1):
        clear_coordinate_grads(batch)
        outputs = model.training_outputs(batch)
        loss, parts = compute_loss(outputs, batch, config["loss"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"XPINN loss became non-finite at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_interval == 0 or step == steps:
            error1, error2, error3, combined_error = evaluate(
                model, evaluation_points, references
            )
            print(
                f"step={step:4d} loss={loss.item():.3e} "
                f"boundary={parts['boundary']:.3e} pde={parts['pde']:.3e} "
                f"interface_r={parts['interface_residual']:.3e} "
                f"interface_u={parts['interface_value']:.3e} "
                f"l2=({error1:.3e}, {error2:.3e}, {error3:.3e}) "
                f"combined={combined_error:.3e}"
            )

    weight_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "case": "poisson2d",
        "architecture": "xpinn_poisson2d",
        "model_state": model.state_dict(),
        "model_config": config["model"],
        "sample_counts": counts,
        "step": steps,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Training finished in {time.time() - started:.1f}s")
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
