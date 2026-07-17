from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS

from onescience.utils.pinnsformer_util import get_data, get_n_params, make_time_sequence

from common import (
    build_model,
    ensure_runtime_dirs,
    initial_condition,
    load_config,
    project_path,
    seed_everything,
    select_device,
)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


def tensorize(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, requires_grad=True, device=device)


def prepare_tensors(cfg: dict, device: torch.device, args: argparse.Namespace):
    data_cfg = cfg["data"]
    x_num = int(args.x_num or data_cfg["x_num"])
    t_num = int(args.t_num or data_cfg["t_num"])
    num_step = int(args.num_step or data_cfg["sequence"]["num_step"])
    step = float(data_cfg["sequence"]["step"])

    res, b_left, b_right, b_upper, b_lower = get_data(
        data_cfg["x_range"],
        data_cfg["t_range"],
        x_num,
        t_num,
    )

    tensors = []
    for values in (res, b_left, b_right, b_upper, b_lower):
        tensors.append(tensorize(make_time_sequence(values, num_step=num_step, step=step), device))

    return tuple(tensors)


def loss_components(model: nn.Module, tensors: tuple[torch.Tensor, ...], cfg: dict):
    res, b_left, b_right, b_upper, b_lower = tensors
    x_res, t_res = res[:, :, 0:1], res[:, :, 1:2]
    x_left, t_left = b_left[:, :, 0:1], b_left[:, :, 1:2]
    x_upper, t_upper = b_upper[:, :, 0:1], b_upper[:, :, 1:2]
    x_lower, t_lower = b_lower[:, :, 0:1], b_lower[:, :, 1:2]

    pred_res = model(x_res, t_res)
    pred_left = model(x_left, t_left)
    pred_upper = model(x_upper, t_upper)
    pred_lower = model(x_lower, t_lower)

    u_t = torch.autograd.grad(
        pred_res,
        t_res,
        grad_outputs=torch.ones_like(pred_res),
        retain_graph=True,
        create_graph=True,
    )[0]

    rate = float(cfg["equation"]["reaction_rate"])
    target_ic = initial_condition(x_left[:, 0, :], cfg)
    loss_res = torch.mean((u_t - rate * pred_res * (1 - pred_res)) ** 2)
    loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
    loss_ic = torch.mean((pred_left[:, 0, :] - target_ic) ** 2)
    loss = loss_res + loss_bc + loss_ic
    return loss, (loss_res, loss_bc, loss_ic)


def build_optimizer(model: nn.Module, cfg: dict):
    opt_cfg = cfg["training"]["optimizer"]
    name = opt_cfg["name"].lower()
    if name == "adam":
        return Adam(model.parameters(), lr=float(opt_cfg.get("lr", 1e-3)))
    if name == "lbfgs":
        return LBFGS(
            model.parameters(),
            lr=float(opt_cfg.get("lr", 1.0)),
            max_iter=int(opt_cfg.get("max_iter", 20)),
            line_search_fn=opt_cfg.get("line_search_fn", "strong_wolfe"),
        )
    raise ValueError(f"Unsupported optimizer: {opt_cfg['name']}")


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, loss_history: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "loss_history": loss_history,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PINNsformer on the 1D reaction equation.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--x-num", type=int, default=None, help="Override x grid count.")
    parser.add_argument("--t-num", type=int, default=None, help="Override t grid count.")
    parser.add_argument("--num-step", type=int, default=None, help="Override pseudo-sequence length.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_runtime_dirs(cfg)
    seed_everything(int(cfg["runtime"]["seed"]))
    device = select_device(args.device or cfg["runtime"]["device"])

    tensors = prepare_tensors(cfg, device, args)
    model = build_model(cfg).to(device)
    model.apply(init_weights)
    optimizer = build_optimizer(model, cfg)
    epochs = int(args.epochs or cfg["training"]["epochs"])

    print(model)
    print(f"parameters: {get_n_params(model)}")
    print(f"device: {device}")

    loss_history: list[list[float]] = []
    for epoch in range(epochs):
        latest: dict[str, float] = {}

        def closure():
            loss, parts = loss_components(model, tensors, cfg)
            optimizer.zero_grad()
            loss.backward()
            latest["loss"] = float(loss.detach().cpu())
            latest["loss_res"] = float(parts[0].detach().cpu())
            latest["loss_bc"] = float(parts[1].detach().cpu())
            latest["loss_ic"] = float(parts[2].detach().cpu())
            return loss

        if isinstance(optimizer, LBFGS):
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()

        loss_history.append([latest["loss_res"], latest["loss_bc"], latest["loss_ic"], latest["loss"]])
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"loss={latest['loss']:.6f} "
            f"res={latest['loss_res']:.6f} "
            f"bc={latest['loss_bc']:.6f} "
            f"ic={latest['loss_ic']:.6f}"
        )

    checkpoint = project_path(cfg["training"]["checkpoint"])
    save_checkpoint(checkpoint, model, cfg, loss_history)
    np.save(project_path(cfg["paths"]["loss"]), np.asarray(loss_history, dtype=np.float32))
    print(f"checkpoint saved to {checkpoint}")


if __name__ == "__main__":
    main()
