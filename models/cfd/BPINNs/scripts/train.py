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
    build_laplace_data,
    load_config,
    project_path,
    relative_l2,
    resolve_device,
    resolve_dtype,
    seed_everything,
)
from model.bpinn import (  # noqa: E402
    build_model,
    laplace1d_loss_components,
    weighted_loss,
)


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    common = config["common"]
    device = resolve_device(str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    seed = int(common["seed"])
    epochs = int(config["training"]["epochs"])
    lbfgs_iters = int(config["training"]["lbfgs_iters"])
    learning_rate = float(config["training"]["lr"])
    if min(epochs, lbfgs_iters) < 0 or epochs + lbfgs_iters == 0:
        raise ValueError("at least one non-negative optimizer iteration count must be positive")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    data_config = dict(config["data"])
    weight_dir = project_path(common["weight_dir"], PROJECT_ROOT)
    result_dir = project_path(common["result_dir"], PROJECT_ROOT)
    checkpoint_path = weight_dir / config["training"]["checkpoint_name"]
    history_path = result_dir / config["inference"]["history_name"]
    seed_everything(seed)

    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Data config: {data_config}")
    data = build_laplace_data(data_config, seed, device, dtype)

    with torch.enable_grad():
        test_points = data["x_test"].detach().requires_grad_(True)
        exact = torch.sin(torch.pi * test_points)
        first = torch.autograd.grad(
            exact, test_points, torch.ones_like(exact), create_graph=True
        )[0]
        second = torch.autograd.grad(
            first, test_points, torch.ones_like(first), create_graph=True
        )[0]
        max_residual = torch.max(
            torch.abs(second + torch.pi**2 * torch.sin(torch.pi * test_points))
        ).item()
    if max_residual > 1.0e-8:
        raise RuntimeError(f"Laplace autograd validation failed: {max_residual:.3e}")
    print(f"PDE autograd validation: {max_residual:.3e}")

    model = build_model(config["model"], dtype=dtype).to(device=device, dtype=dtype)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count:,}")

    def loss_value() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = laplace1d_loss_components(
            model,
            data["x_solution"],
            data["u_solution"],
            data["x_boundary"],
            data["u_boundary"],
            data["x_pde"],
        )
        return weighted_loss(components, config["loss"]), components

    def evaluate() -> float:
        with torch.no_grad():
            prediction = model.predict_u(data["x_test"])
        return relative_l2(prediction, data["u_test"])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    l2_history = []
    log_interval = int(config["training"]["log_interval"])
    best_l2 = evaluate()
    started = time.time()
    print(
        f"Adam epochs={epochs} lr={learning_rate:g}, "
        f"L-BFGS iterations={lbfgs_iters}"
    )
    for epoch in range(1, epochs + 1):
        loss, components = loss_value()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"BPINN loss became non-finite at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            error = evaluate()
            l2_history.append((epoch, error))
            best_l2 = min(best_l2, error)
            print(
                f"epoch={epoch:6d} loss={loss.item():.3e} "
                f"data={components['data'].item():.3e} "
                f"boundary={components['boundary'].item():.3e} "
                f"pde={components['pde'].item():.3e} l2={error:.3e}"
            )

    if lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_iters,
            max_eval=max(1, 2 * lbfgs_iters),
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss, _ = loss_value()
            if not torch.isfinite(closure_loss):
                raise FloatingPointError("BPINN L-BFGS loss became non-finite")
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
        "case": "laplace1d",
        "architecture": "bpinn",
        "model_state": model.state_dict(),
        "model_config": config["model"],
        "data_config": data_config,
        "loss_weights": config["loss"],
        "seed": seed,
        "epochs": epochs,
        "lbfgs_iters": lbfgs_iters,
        "final_l2": final_l2,
    }
    torch.save(checkpoint, checkpoint_path)
    l2_array = np.asarray(l2_history, dtype=np.float64).reshape(-1, 2)
    np.savez_compressed(
        history_path,
        loss=np.asarray(loss_history, dtype=np.float64),
        l2_steps=l2_array[:, 0],
        l2_values=l2_array[:, 1],
        final_l2=final_l2,
        best_l2=best_l2,
        elapsed_seconds=elapsed,
        parameter_count=parameter_count,
    )
    print(f"Final relative L2={final_l2:.6e}, elapsed={elapsed:.1f}s")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
