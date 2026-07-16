from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_config,
    project_path,
    relative_l2,
    resolve_device,
    resolve_dtype,
    seed_everything,
)
from data_utils import batched_predict, load_allen_cahn, sample_training_data  # noqa: E402
from model.fpinn import build_model, loss_components, weighted_loss  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"
TASKS = ("forward", "inverse")


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    common = config["common"]
    task = str(common["task"]).lower()
    if task not in TASKS:
        raise ValueError("common.task must be one of: forward, inverse")
    task_config = config["tasks"][task]
    device = resolve_device(str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    seed = int(common["seed"])
    epochs = int(task_config["training"]["epochs"])
    lbfgs_iters = int(task_config["training"]["lbfgs_iters"])
    learning_rate = float(task_config["training"]["lr"])
    if min(epochs, lbfgs_iters) < 0 or epochs + lbfgs_iters == 0:
        raise ValueError("at least one optimizer iteration count must be positive")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    data_config = dict(config["data"])
    data_path = project_path(data_config["mat_file"], PROJECT_ROOT)
    weight_dir = project_path(common["weight_dir"], PROJECT_ROOT)
    result_dir = project_path(common["result_dir"], PROJECT_ROOT)
    checkpoint_path = weight_dir / task_config["output"]["checkpoint_name"]
    history_path = result_dir / task_config["output"]["history_name"]
    seed_everything(seed)

    dataset = load_allen_cahn(data_path)
    x_train, u_train = sample_training_data(
        dataset,
        int(data_config["n_train"]),
        seed,
        device,
        dtype,
    )
    model = build_model(task, config["model"], task_config["pde"], dtype).to(
        device=device, dtype=dtype
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if task == "forward":
        lambda_1 = float(task_config["pde"]["lambda_1"])
        lambda_2 = float(task_config["pde"]["lambda_2"])
    else:
        lambda_1 = model.lambda_1
        lambda_2 = model.lambda_2

    print(f"Config: {config_path}")
    print(f"Task: {task}")
    print(f"Data: {data_path}")
    print(f"Device: {device}")
    print(f"Training points: {x_train.shape[0]}")
    print(f"Parameters: {parameter_count:,}")

    with torch.enable_grad():
        sample = x_train[: min(10, x_train.shape[0])].detach().requires_grad_(True)
        test_function = sample[:, 0:1].square() + sample[:, 1:2]
        gradient = torch.autograd.grad(
            test_function,
            sample,
            torch.ones_like(test_function),
            create_graph=True,
        )[0]
        second = torch.autograd.grad(
            gradient[:, 0:1],
            sample,
            torch.ones_like(gradient[:, 0:1]),
            create_graph=True,
        )[0][:, 0:1]
        if not torch.allclose(gradient[:, 1:2], torch.ones_like(second), atol=1.0e-5):
            raise RuntimeError("u_t autograd validation failed")
        if not torch.allclose(second, torch.full_like(second, 2.0), atol=1.0e-5):
            raise RuntimeError("u_xx autograd validation failed")
    print("Autograd validation: passed")

    def current_loss() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = loss_components(
            model, x_train, u_train, x_train, lambda_1, lambda_2
        )
        return weighted_loss(components, task_config["loss"]), components

    def evaluate() -> float:
        prediction = batched_predict(
            model,
            np.asarray(dataset["coordinates"]),
            int(data_config["evaluation_batch_size"]),
            device,
            dtype,
        )
        return relative_l2(prediction, np.asarray(dataset["exact"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log_interval = int(task_config["training"]["log_interval"])
    loss_history = []
    l2_history = []
    lambda_history = []
    best_l2 = evaluate()
    started = time.time()
    for epoch in range(1, epochs + 1):
        loss, components = current_loss()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"FPINN loss became non-finite at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if task == "inverse":
            lambda_history.append((model.lambda_1.item(), model.lambda_2.item()))

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            error = evaluate()
            l2_history.append((epoch, error))
            best_l2 = min(best_l2, error)
            lambda_text = (
                ""
                if task == "forward"
                else f" lambda=({model.lambda_1.item():.6e}, {model.lambda_2.item():.6e})"
            )
            print(
                f"epoch={epoch:6d} loss={loss.item():.3e} "
                f"data={components['data'].item():.3e} "
                f"pde={components['pde'].item():.3e} l2={error:.3e}{lambda_text}"
            )

    if lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_iters,
            max_eval=max(1, 2 * lbfgs_iters),
            history_size=50,
            tolerance_grad=1.0e-5,
            tolerance_change=float(np.finfo(float).eps),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss, _ = current_loss()
            if not torch.isfinite(closure_loss):
                raise FloatingPointError("FPINN L-BFGS loss became non-finite")
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)
        error = evaluate()
        l2_history.append((epochs + lbfgs_iters, error))
        best_l2 = min(best_l2, error)
        print(f"L-BFGS relative L2={error:.6e}")

    final_l2 = evaluate()
    elapsed = time.time() - started
    weight_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "task": task,
        "case": "allen_cahn",
        "architecture": "fuzzy_pinn",
        "model_state": model.state_dict(),
        "model_config": config["model"],
        "pde_config": task_config["pde"],
        "data_config": data_config,
        "seed": seed,
        "epochs": epochs,
        "lbfgs_iters": lbfgs_iters,
        "final_l2": final_l2,
    }
    torch.save(checkpoint, checkpoint_path)
    l2_array = np.asarray(l2_history, dtype=np.float64).reshape(-1, 2)
    lambda_array = np.asarray(lambda_history, dtype=np.float64).reshape(-1, 2)
    np.savez_compressed(
        history_path,
        loss=np.asarray(loss_history, dtype=np.float64),
        l2_steps=l2_array[:, 0],
        l2_values=l2_array[:, 1],
        lambda_1=lambda_array[:, 0] if lambda_array.size else np.array([]),
        lambda_2=lambda_array[:, 1] if lambda_array.size else np.array([]),
        final_l2=final_l2,
        best_l2=best_l2,
        elapsed_seconds=elapsed,
        parameter_count=parameter_count,
    )
    print(f"Final relative L2={final_l2:.6e}, elapsed={elapsed:.1f}s")
    if task == "inverse":
        print(
            f"Recovered lambda_1={model.lambda_1.item():.8e}, "
            f"lambda_2={model.lambda_2.item():.8e}"
        )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
