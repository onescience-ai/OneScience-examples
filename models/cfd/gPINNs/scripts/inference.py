from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.gpinn import (  # noqa: E402
    exact_poisson1d,
    exact_poisson2d,
    gPINN,
    gPINNBurgers,
    gPINNPoisson2D,
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


def load_checkpoint(weight_dir: Path, filename: str) -> dict:
    checkpoint_path = weight_dir / filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"invalid checkpoint format: {checkpoint_path}")
    return checkpoint


def infer_poisson1d(
    weight_dir: Path,
    result_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    checkpoint = load_checkpoint(weight_dir, "gpinn_poisson1d.pt")
    layers = checkpoint.get("layers", [1, 30, 30, 30, 30, 1])
    model = gPINN(layers).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    x = torch.linspace(0.0, np.pi, 1000, dtype=dtype, device=device).unsqueeze(-1)
    exact = exact_poisson1d(x.cpu().numpy().reshape(-1))
    with torch.no_grad():
        prediction = model(x).cpu().numpy().reshape(-1)
    relative_l2 = float(np.linalg.norm(prediction - exact) / np.linalg.norm(exact))

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(x.cpu().numpy(), exact, "k-", label="Exact")
    axis.plot(x.cpu().numpy(), prediction, "r--", label="gPINN")
    axis.set_xlabel("x")
    axis.set_ylabel("u")
    axis.legend()
    figure.tight_layout()
    output_path = result_dir / "gpinn_poisson1d.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"[1D] relative L2={relative_l2:.6e}, plot={output_path}")
    return relative_l2


def infer_poisson2d(
    weight_dir: Path,
    result_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    default_exponent: float,
) -> float:
    checkpoint = load_checkpoint(weight_dir, "gpinn_poisson2d.pt")
    layers = checkpoint.get("layers", [2, 30, 30, 30, 1])
    architecture = checkpoint.get("architecture", "gpinn")
    if architecture == "poisson2d_hard_bc":
        model = gPINNPoisson2D(layers)
    elif architecture == "gpinn":
        # The bundled legacy checkpoint predates the hard boundary transform.
        model = gPINN(layers)
    else:
        raise ValueError(f"unsupported 2D Poisson architecture: {architecture}")
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    exponent = float(checkpoint.get("a", default_exponent))
    axis = np.linspace(0.0, 1.0, 100)
    grid_x, grid_y = np.meshgrid(axis, axis)
    coordinates = torch.as_tensor(
        np.column_stack([grid_x.ravel(), grid_y.ravel()]), dtype=dtype, device=device
    )
    with torch.no_grad():
        prediction = model(coordinates).cpu().numpy().reshape(grid_x.shape)
    exact = exact_poisson2d(grid_x, grid_y, exponent)
    absolute_error = np.abs(prediction - exact)
    relative_l2 = float(
        np.linalg.norm(prediction.ravel() - exact.ravel()) / np.linalg.norm(exact.ravel())
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for plot_axis, title, field in zip(
        axes,
        ("Exact", "gPINN", "Absolute error"),
        (exact, prediction, absolute_error),
        strict=True,
    ):
        image = plot_axis.imshow(field, extent=(0, 1, 0, 1), origin="lower", cmap="jet")
        plot_axis.set_title(title)
        plot_axis.set_xlabel("x")
        plot_axis.set_ylabel("y")
        figure.colorbar(image, ax=plot_axis)
    figure.tight_layout()
    output_path = result_dir / "gpinn_poisson2d.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"[2D] relative L2={relative_l2:.6e}, plot={output_path}")
    return relative_l2


def load_burgers_grid(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return time_axis, space_axis, exact


def infer_burgers(
    weight_dir: Path,
    result_dir: Path,
    data_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    checkpoint = load_checkpoint(weight_dir, "gpinn_burgers.pt")
    layers = checkpoint.get("layers", [2, 32, 32, 32, 1])
    model = gPINNBurgers(layers).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    time_axis, space_axis, exact = load_burgers_grid(data_path)
    time_grid, space_grid = np.meshgrid(time_axis, space_axis)
    coordinates = torch.as_tensor(
        np.column_stack([space_grid.ravel(), time_grid.ravel()]),
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        prediction = model(coordinates).cpu().numpy().reshape(exact.shape)
    absolute_error = np.abs(prediction - exact)
    relative_l2 = float(
        np.linalg.norm(prediction.ravel() - exact.ravel()) / np.linalg.norm(exact.ravel())
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for plot_axis, title, field in zip(
        axes,
        ("Exact", "gPINN", "Absolute error"),
        (exact, prediction, absolute_error),
        strict=True,
    ):
        image = plot_axis.contourf(time_axis, space_axis, field, 100, cmap="jet")
        plot_axis.set_title(title)
        plot_axis.set_xlabel("t")
        plot_axis.set_ylabel("x")
        figure.colorbar(image, ax=plot_axis)
    figure.tight_layout()
    output_path = result_dir / "gpinn_burgers.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"[Burgers] relative L2={relative_l2:.6e}, plot={output_path}")
    return relative_l2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gPINN inference and visualization")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("GPINN_CONFIG", DEFAULT_CONFIG)),
        help="YAML configuration file",
    )
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--device", help="Override common.device, for example cpu or cuda:0")
    parser.add_argument("--data", type=Path, help="Override Burgers.npz path")
    parser.add_argument("--weight-dir", type=Path, help="Override checkpoint input directory")
    parser.add_argument("--result-dir", type=Path, help="Override plot output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    common = config["common"]
    device = resolve_device(args.device or str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    weight_dir = project_path(args.weight_dir or common["weight_dir"])
    result_dir = project_path(args.result_dir or common["result_dir"])
    data_path = project_path(args.data or config["burgers"]["data"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Device: {device}")
    selected_cases = CASES if args.case == "all" else (args.case,)
    if "1d" in selected_cases:
        infer_poisson1d(weight_dir, result_dir, device, dtype)
    if "2d" in selected_cases:
        infer_poisson2d(
            weight_dir,
            result_dir,
            device,
            dtype,
            float(config["poisson2d"]["a"]),
        )
    if "burgers" in selected_cases:
        infer_burgers(weight_dir, result_dir, data_path, device, dtype)


if __name__ == "__main__":
    main()
