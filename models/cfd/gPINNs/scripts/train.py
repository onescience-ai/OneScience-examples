from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpinn import (  # noqa: E402
    burgers_gpinn_terms,
    burgers_residual,
    exact_poisson1d,
    exact_poisson2d,
    gPINN,
    gPINNBurgers,
    gPINNPoisson2D,
    gpinn_loss_poisson1d,
    gpinn_loss_poisson2d,
)


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"
CASES = ("1d", "2d", "burgers")


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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_finite(loss: torch.Tensor, case: str, step: int) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{case} loss became non-finite at step {step}")


def save_checkpoint(weight_dir: Path, filename: str, payload: dict) -> Path:
    weight_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weight_dir / filename
    torch.save(payload, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def train_poisson1d(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    weight_dir: Path,
    epochs_override: int | None,
    nf_override: int | None,
    lbfgs_override: int | None,
) -> None:
    layers = [int(value) for value in config["layers"]]
    nf = nf_override if nf_override is not None else int(config["nf"])
    epochs = epochs_override if epochs_override is not None else int(config["epochs"])
    lbfgs_iters = (
        lbfgs_override if lbfgs_override is not None else int(config["lbfgs_iters"])
    )
    log_every = int(config["log_every"])

    interior = torch.linspace(0.0, np.pi, nf, dtype=dtype, device=device).unsqueeze(-1)
    boundary = torch.tensor([[0.0], [np.pi]], dtype=dtype, device=device)
    boundary_values = boundary.clone()
    test_x = torch.linspace(0.0, np.pi, 1000, dtype=dtype, device=device).unsqueeze(-1)
    exact = torch.as_tensor(
        exact_poisson1d(test_x.detach().cpu().numpy().reshape(-1)),
        dtype=dtype,
        device=device,
    ).unsqueeze(-1)

    model = gPINN(layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    started = time.time()
    print(
        f"[1D] device={device} nf={nf} epochs={epochs} "
        f"lbfgs_iters={lbfgs_iters}"
    )

    for step in range(1, epochs + 1):
        loss, parts = gpinn_loss_poisson1d(
            model, interior, boundary, boundary_values, float(config["w_g"])
        )
        ensure_finite(loss, "1D Poisson", step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % log_every == 0 or step == epochs:
            with torch.no_grad():
                relative_l2 = torch.linalg.vector_norm(model(test_x) - exact) / torch.linalg.vector_norm(
                    exact
                )
            print(
                f"[1D] step={step:5d} loss={loss.item():.3e} "
                f"residual={parts['residual']:.3e} boundary={parts['boundary']:.3e} "
                f"gradient={parts['gradient']:.3e} l2={relative_l2.item():.3e}"
            )

    if lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_iters,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss, _ = gpinn_loss_poisson1d(
                model, interior, boundary, boundary_values, float(config["w_g"])
            )
            ensure_finite(closure_loss, "1D Poisson L-BFGS", 0)
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)

    with torch.no_grad():
        relative_l2 = torch.linalg.vector_norm(model(test_x) - exact) / torch.linalg.vector_norm(exact)
    print(f"[1D] finished in {time.time() - started:.1f}s, relative L2={relative_l2.item():.6e}")
    save_checkpoint(
        weight_dir,
        "gpinn_poisson1d.pt",
        {
            "case": "1d",
            "architecture": "gpinn",
            "model_state": model.state_dict(),
            "layers": layers,
            "nf": nf,
            "w_g": float(config["w_g"]),
        },
    )


def train_poisson2d(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    weight_dir: Path,
    epochs_override: int | None,
    nf_override: int | None,
) -> None:
    layers = [int(value) for value in config["layers"]]
    nf = nf_override if nf_override is not None else int(config["nf"])
    epochs = epochs_override if epochs_override is not None else int(config["epochs"])
    exponent = float(config["a"])
    log_every = int(config["log_every"])

    interior = torch.rand(nf, 2, dtype=dtype, device=device)
    model = gPINNPoisson2D(layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    started = time.time()
    print(f"[2D] device={device} nf={nf} epochs={epochs} a={exponent:g}")

    for step in range(1, epochs + 1):
        loss, parts = gpinn_loss_poisson2d(
            model, interior, float(config["w_g"]), exponent
        )
        ensure_finite(loss, "2D Poisson", step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % log_every == 0 or step == epochs:
            print(
                f"[2D] step={step:5d} loss={loss.item():.3e} "
                f"residual={parts['residual']:.3e} gradient={parts['gradient']:.3e}"
            )

    axis = np.linspace(0.0, 1.0, 100)
    grid_x, grid_y = np.meshgrid(axis, axis)
    test_coordinates = torch.as_tensor(
        np.column_stack([grid_x.ravel(), grid_y.ravel()]), dtype=dtype, device=device
    )
    with torch.no_grad():
        prediction = model(test_coordinates).cpu().numpy().reshape(grid_x.shape)
    exact = exact_poisson2d(grid_x, grid_y, exponent)
    relative_l2 = np.linalg.norm(prediction.ravel() - exact.ravel()) / np.linalg.norm(
        exact.ravel()
    )
    print(f"[2D] finished in {time.time() - started:.1f}s, relative L2={relative_l2:.6e}")
    save_checkpoint(
        weight_dir,
        "gpinn_poisson2d.pt",
        {
            "case": "2d",
            "architecture": "poisson2d_hard_bc",
            "model_state": model.state_dict(),
            "layers": layers,
            "a": exponent,
            "nf": nf,
            "w_g": float(config["w_g"]),
        },
    )


def load_burgers_reference(
    data_path: Path, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Burgers data not found: {data_path}")
    with np.load(data_path) as data:
        missing = {"t", "x", "usol"}.difference(data.files)
        if missing:
            raise ValueError(f"Burgers data is missing arrays: {sorted(missing)}")
        time_axis = np.asarray(data["t"]).reshape(-1)
        space_axis = np.asarray(data["x"]).reshape(-1)
        exact = np.asarray(data["usol"])
    expected_shape = (space_axis.size, time_axis.size)
    if exact.shape != expected_shape:
        raise ValueError(f"usol shape must be {expected_shape}, got {exact.shape}")
    time_grid, space_grid = np.meshgrid(time_axis, space_axis)
    coordinates = torch.as_tensor(
        np.column_stack([space_grid.ravel(), time_grid.ravel()]),
        dtype=dtype,
        device=device,
    )
    values = torch.as_tensor(exact.reshape(-1, 1), dtype=dtype, device=device)
    return coordinates, values


def burgers_boundary_points(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boundary = torch.cat(
        [
            torch.cat(
                [
                    torch.full((100, 1), -1.0, dtype=dtype, device=device),
                    torch.rand(100, 1, dtype=dtype, device=device),
                ],
                dim=1,
            ),
            torch.cat(
                [
                    torch.full((100, 1), 1.0, dtype=dtype, device=device),
                    torch.rand(100, 1, dtype=dtype, device=device),
                ],
                dim=1,
            ),
        ]
    )
    initial = torch.cat(
        [
            torch.linspace(-1.0, 1.0, 200, dtype=dtype, device=device).unsqueeze(-1),
            torch.zeros(200, 1, dtype=dtype, device=device),
        ],
        dim=1,
    )
    initial_values = -torch.sin(torch.pi * initial[:, 0:1])
    return boundary, initial, initial_values


def add_rar_points(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    candidate_count = int(config["rar_candidate_points"])
    batch_size = int(config["rar_candidate_batch_size"])
    add_count = int(config["rar_add_points"])
    if min(candidate_count, batch_size, add_count) <= 0:
        raise ValueError("RAR point counts must be positive")

    shortlisted_points = []
    shortlisted_residuals = []
    was_training = model.training
    model.eval()
    for start in range(0, candidate_count, batch_size):
        current_size = min(batch_size, candidate_count - start)
        candidates = torch.rand(current_size, 2, dtype=dtype, device=device)
        candidates[:, 0] = 2.0 * candidates[:, 0] - 1.0
        residual, inputs = burgers_residual(model, candidates, create_graph=False)
        count = min(add_count, current_size)
        values, indices = torch.topk(residual.detach().abs().reshape(-1), count)
        shortlisted_points.append(inputs.detach()[indices])
        shortlisted_residuals.append(values)
    if was_training:
        model.train()

    residuals = torch.cat(shortlisted_residuals)
    points = torch.cat(shortlisted_points)
    final_count = min(add_count, residuals.numel())
    indices = torch.topk(residuals, final_count).indices
    return points[indices].detach()


def train_burgers(
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
    weight_dir: Path,
    data_path: Path,
    epochs_override: int | None,
    nf_override: int | None,
    rar_override: int | None,
    quick: bool,
) -> None:
    layers = [int(value) for value in config["layers"]]
    nf = nf_override if nf_override is not None else int(config["nf"])
    epochs = epochs_override if epochs_override is not None else int(config["epochs"])
    rar_rounds = rar_override if rar_override is not None else int(config["rar_rounds"])
    if quick:
        rar_rounds = 0
    log_every = int(config["log_every"])

    collocation = torch.rand(nf, 2, dtype=dtype, device=device)
    collocation[:, 0] = 2.0 * collocation[:, 0] - 1.0
    reference_coordinates, reference_values = load_burgers_reference(data_path, device, dtype)
    boundary, initial, initial_values = burgers_boundary_points(device, dtype)
    model = gPINNBurgers(layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    anchors: list[torch.Tensor] = []

    def evaluate() -> float:
        with torch.no_grad():
            return (
                torch.linalg.vector_norm(model(reference_coordinates) - reference_values)
                / torch.linalg.vector_norm(reference_values)
            ).item()

    def optimization_step(points: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        residual, residual_x, residual_t = burgers_gpinn_terms(model, points)
        residual_loss = torch.mean(residual.square())
        boundary_loss = torch.mean(model(boundary).square()) + torch.mean(
            (model(initial) - initial_values).square()
        )
        gradient_loss = torch.mean(residual_x.square()) + torch.mean(residual_t.square())
        loss = residual_loss + boundary_loss + float(config["w_g"]) * gradient_loss
        ensure_finite(loss, "Burgers", step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss, {
            "residual": residual_loss.item(),
            "boundary": boundary_loss.item(),
            "gradient": gradient_loss.item(),
        }

    started = time.time()
    print(f"[Burgers] device={device} nf={nf} epochs={epochs} rar_rounds={rar_rounds}")
    for step in range(1, epochs + 1):
        points = torch.cat([collocation, *anchors], dim=0)
        loss, parts = optimization_step(points, step)
        if step == 1 or step % log_every == 0 or step == epochs:
            print(
                f"[Burgers] step={step:5d} loss={loss.item():.3e} "
                f"residual={parts['residual']:.3e} boundary={parts['boundary']:.3e} "
                f"gradient={parts['gradient']:.3e} l2={evaluate():.3e}"
            )

    rar_epochs = int(config["rar_epochs"])
    for round_index in range(1, rar_rounds + 1):
        new_points = add_rar_points(model, config, device, dtype)
        anchors.append(new_points)
        points = torch.cat([collocation, *anchors], dim=0)
        for rar_step in range(1, rar_epochs + 1):
            optimization_step(points, epochs + (round_index - 1) * rar_epochs + rar_step)
        print(
            f"[Burgers] RAR round={round_index}/{rar_rounds} "
            f"anchors={sum(item.shape[0] for item in anchors)} l2={evaluate():.3e}"
        )

    relative_l2 = evaluate()
    print(
        f"[Burgers] finished in {time.time() - started:.1f}s, "
        f"relative L2={relative_l2:.6e}"
    )
    save_checkpoint(
        weight_dir,
        "gpinn_burgers.pt",
        {
            "case": "burgers",
            "architecture": "burgers_hard_bc",
            "model_state": model.state_dict(),
            "layers": layers,
            "nf": nf,
            "w_g": float(config["w_g"]),
            "rar_rounds": rar_rounds,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gPINN PDE examples")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("GPINN_CONFIG", DEFAULT_CONFIG)),
        help="YAML configuration file",
    )
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--device", help="Override common.device, for example cpu or cuda:0")
    parser.add_argument("--epochs", type=int, help="Override Adam iterations for selected cases")
    parser.add_argument("--nf", type=int, help="Override collocation point count")
    parser.add_argument("--lbfgs-iters", type=int, help="Override 1D Poisson L-BFGS iterations")
    parser.add_argument("--rar-rounds", type=int, help="Override Burgers RAR rounds")
    parser.add_argument("--quick", action="store_true", help="Skip Burgers RAR rounds")
    parser.add_argument("--data", type=Path, help="Override Burgers.npz path")
    parser.add_argument("--weight-dir", type=Path, help="Override checkpoint output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    common = config["common"]
    device = resolve_device(args.device or str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    weight_dir = project_path(args.weight_dir or common["weight_dir"])
    data_path = project_path(args.data or config["burgers"]["data"])
    seed_everything(int(common["seed"]))

    print(f"Config: {config_path}")
    selected_cases = CASES if args.case == "all" else (args.case,)
    if "1d" in selected_cases:
        train_poisson1d(
            config["poisson1d"],
            device,
            dtype,
            weight_dir,
            args.epochs,
            args.nf,
            args.lbfgs_iters,
        )
    if "2d" in selected_cases:
        train_poisson2d(
            config["poisson2d"],
            device,
            dtype,
            weight_dir,
            args.epochs,
            args.nf,
        )
    if "burgers" in selected_cases:
        train_burgers(
            config["burgers"],
            device,
            dtype,
            weight_dir,
            data_path,
            args.epochs,
            args.nf,
            args.rar_rounds,
            args.quick,
        )


if __name__ == "__main__":
    main()
