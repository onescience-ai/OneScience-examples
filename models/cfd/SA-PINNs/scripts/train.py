from __future__ import annotations

import argparse
import os
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
    resolve_device,
    resolve_dtype,
    seed_everything,
)
from model.sa_pinn import build_model, loss_components, weighted_loss  # noqa: E402
from problems import (  # noqa: E402
    CASES,
    build_equation,
    generate_data,
    point_counts,
    relative_l2,
    to_tensors,
)


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-adaptive PINN")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("SAPINN_CONFIG", DEFAULT_CONFIG)),
        help="YAML configuration file",
    )
    parser.add_argument("--case", choices=CASES, default="laplace")
    parser.add_argument("--device", help="Override common.device, for example cpu or cuda:0")
    parser.add_argument("--seed", type=int, help="Override common.seed")
    parser.add_argument("--epochs", type=int, help="Override Adam iterations")
    parser.add_argument("--lr", type=float, help="Override network learning rate")
    parser.add_argument("--attention-lr", type=float, help="Override attention learning rate")
    parser.add_argument("--lbfgs-iters", type=int, help="Override L-BFGS iterations")
    parser.add_argument("--n-sol", type=int, help="Override data or initial-condition points")
    parser.add_argument("--n-pde", type=int, help="Override PDE collocation points")
    parser.add_argument("--n-bnd", type=int, help="Override boundary points")
    parser.add_argument("--test-res", type=int, help="Override test resolution per axis")
    parser.add_argument("--weight-dir", type=Path, help="Override checkpoint output directory")
    parser.add_argument("--result-dir", type=Path, help="Override history output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    common = config["common"]
    case_config = config["cases"][args.case]
    device = resolve_device(args.device or str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    seed = args.seed if args.seed is not None else int(common["seed"])
    epochs = (
        args.epochs if args.epochs is not None else int(case_config["training"]["epochs"])
    )
    lbfgs_iters = (
        args.lbfgs_iters
        if args.lbfgs_iters is not None
        else int(case_config["training"]["lbfgs_iters"])
    )
    network_lr = (
        args.lr if args.lr is not None else float(case_config["training"]["lr"])
    )
    attention_lr = (
        args.attention_lr
        if args.attention_lr is not None
        else float(case_config["attention"]["lr"])
    )
    if min(epochs, lbfgs_iters) < 0 or epochs + lbfgs_iters == 0:
        raise ValueError("at least one optimizer iteration count must be positive")
    if min(network_lr, attention_lr) <= 0:
        raise ValueError("network and attention learning rates must be positive")

    data_config = dict(case_config["data"])
    for key, value in (
        ("n_sol", args.n_sol),
        ("n_pde", args.n_pde),
        ("n_bnd", args.n_bnd),
        ("test_res", args.test_res),
    ):
        if value is not None:
            data_config[key] = value
    weight_dir = project_path(args.weight_dir or common["weight_dir"], PROJECT_ROOT)
    result_dir = project_path(args.result_dir or common["result_dir"], PROJECT_ROOT)
    checkpoint_path = weight_dir / case_config["output"]["checkpoint_name"]
    history_path = result_dir / case_config["output"]["history_name"]
    seed_everything(seed)

    data = generate_data(args.case, data_config, seed)
    counts = point_counts(data)
    tensors = to_tensors(data, device, dtype)
    attention_enabled = bool(case_config["attention"]["enabled"])
    model = build_model(
        case_config["model"], counts, attention_enabled, dtype
    ).to(device=device, dtype=dtype)
    equation = build_equation(args.case, data_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Config: {config_path}")
    print(f"Case: {args.case}")
    print(f"Device: {device}")
    print(f"Points: {counts}")
    print(f"Parameters: {parameter_count:,}")

    def loss_value() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = loss_components(model, equation, tensors)
        return weighted_loss(components, case_config["loss"]), components

    def evaluate() -> float | None:
        with torch.no_grad():
            prediction = model(tensors["x_test"]).cpu().numpy()
        return relative_l2(prediction, data["u_exact"])

    network_optimizer = torch.optim.Adam(model.network_parameters(), lr=network_lr)
    attention_parameters = list(model.attention_parameters())
    attention_optimizer = (
        torch.optim.Adam(attention_parameters, lr=attention_lr)
        if attention_parameters
        else None
    )
    history = []
    l2_history = []
    log_interval = int(case_config["training"]["log_interval"])
    started = time.time()
    for epoch in range(1, epochs + 1):
        network_optimizer.zero_grad(set_to_none=True)
        if attention_optimizer is not None:
            attention_optimizer.zero_grad(set_to_none=True)
        loss, components = loss_value()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"SA-PINN loss became non-finite at epoch {epoch}")
        loss.backward()
        network_optimizer.step()
        if attention_optimizer is not None:
            for parameter in attention_parameters:
                if parameter.grad is not None:
                    parameter.grad.neg_()
            attention_optimizer.step()
        history.append(loss.item())

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            error = evaluate()
            if error is not None:
                l2_history.append((epoch, error))
            attention = model.att_pde()
            error_text = "N/A" if error is None else f"{error:.3e}"
            print(
                f"epoch={epoch:6d} loss={loss.item():.3e} "
                f"data={components['data'].item():.3e} "
                f"boundary={components['boundary'].item():.3e} "
                f"pde={components['pde'].item():.3e} l2={error_text} "
                f"attention_std={attention.std(unbiased=False).item():.3e}"
            )

    if lbfgs_iters > 0:
        model.set_attention_trainable(False)
        lbfgs = torch.optim.LBFGS(
            model.network_parameters(),
            lr=0.8,
            max_iter=lbfgs_iters,
            max_eval=max(1, 2 * lbfgs_iters),
            tolerance_grad=1.0e-7,
            tolerance_change=1.0e-9,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss, _ = loss_value()
            if not torch.isfinite(closure_loss):
                raise FloatingPointError("SA-PINN L-BFGS loss became non-finite")
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)
        model.set_attention_trainable(True)
        error = evaluate()
        if error is not None:
            l2_history.append((epochs + lbfgs_iters, error))
        print(f"L-BFGS L2={'N/A' if error is None else f'{error:.6e}'}")

    final_l2 = evaluate()
    elapsed = time.time() - started
    weight_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "case": args.case,
        "architecture": "sa_pinn",
        "model_state": model.state_dict(),
        "model_config": case_config["model"],
        "attention_enabled": attention_enabled,
        "point_counts": counts,
        "data_config": data_config,
        "seed": seed,
        "epochs": epochs,
        "lbfgs_iters": lbfgs_iters,
        "final_l2": final_l2,
    }
    torch.save(checkpoint, checkpoint_path)
    l2_array = np.asarray(l2_history, dtype=np.float64).reshape(-1, 2)
    np.savez_compressed(
        history_path,
        loss=np.asarray(history, dtype=np.float64),
        l2_steps=l2_array[:, 0] if l2_array.size else np.array([]),
        l2_values=l2_array[:, 1] if l2_array.size else np.array([]),
        elapsed_seconds=elapsed,
        final_l2=np.nan if final_l2 is None else final_l2,
    )
    print(f"Final L2={'N/A' if final_l2 is None else f'{final_l2:.6e}'}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
