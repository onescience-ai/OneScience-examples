from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.vpinn import VPINN, infer_layers, unpack_checkpoint  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"
CASES = ("1d", "2d")


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


def load_model(
    checkpoint_path: Path, device: torch.device, dtype: torch.dtype
) -> tuple[VPINN, dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"invalid VPINN checkpoint: {checkpoint_path}")
    state, metadata = unpack_checkpoint(checkpoint)
    layers = metadata.get("layers") or infer_layers(state)
    model = VPINN(layers, dtype=dtype).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata


def exact_poisson1d(x: np.ndarray) -> np.ndarray:
    return 0.1 * np.sin(8.0 * np.pi * x) + np.tanh(80.0 * x)


def infer_poisson1d(
    weight_dir: Path,
    result_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    checkpoint_path = weight_dir / "hpvpinn_poisson1d.pt"
    model, _ = load_model(checkpoint_path, device, dtype)
    points = torch.linspace(-1.0, 1.0, 2001, dtype=dtype, device=device).unsqueeze(-1)
    exact = exact_poisson1d(points.cpu().numpy().reshape(-1))
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(-1)
    absolute_error = np.abs(prediction - exact)
    relative_l2 = float(np.linalg.norm(prediction - exact) / np.linalg.norm(exact))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    coordinates = points.cpu().numpy().reshape(-1)
    axes[0].plot(coordinates, exact, "k-", linewidth=1.5, label="Exact")
    axes[0].plot(coordinates, prediction, "r--", linewidth=1.0, label="hp-VPINN")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend()
    axes[1].semilogy(coordinates, np.maximum(absolute_error, 1.0e-16), "r-")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute error")
    figure.tight_layout()
    output_path = result_dir / "hpvpinn_poisson1d.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"[1D] relative L2={relative_l2:.6e}, plot={output_path}")
    return relative_l2


def exact_poisson2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (0.1 * np.sin(2.0 * np.pi * x) + np.tanh(10.0 * x)) * np.sin(
        2.0 * np.pi * y
    )


def infer_poisson2d(
    weight_dir: Path,
    result_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    checkpoint_path = weight_dir / "hpvpinn_poisson2d.pt"
    model, _ = load_model(checkpoint_path, device, dtype)
    axis = np.linspace(-1.0, 1.0, 100)
    mesh_x, mesh_y = np.meshgrid(axis, axis, indexing="ij")
    points = torch.as_tensor(
        np.column_stack((mesh_x.ravel(), mesh_y.ravel())), dtype=dtype, device=device
    )
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(mesh_x.shape)
    exact = exact_poisson2d(mesh_x, mesh_y)
    absolute_error = np.abs(prediction - exact)
    relative_l2 = float(
        np.linalg.norm(prediction.ravel() - exact.ravel()) / np.linalg.norm(exact.ravel())
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for plot_axis, title, field in zip(
        axes,
        ("Exact", "hp-VPINN", "Absolute error"),
        (exact, prediction, absolute_error),
        strict=True,
    ):
        image = plot_axis.imshow(
            field.T,
            extent=(-1, 1, -1, 1),
            origin="lower",
            aspect="auto",
            cmap="jet",
        )
        plot_axis.set_title(title)
        plot_axis.set_xlabel("x")
        plot_axis.set_ylabel("y")
        figure.colorbar(image, ax=plot_axis)
    figure.tight_layout()
    output_path = result_dir / "hpvpinn_poisson2d.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"[2D] relative L2={relative_l2:.6e}, plot={output_path}")
    return relative_l2


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    common = config["common"]
    device = resolve_device(str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    weight_dir = project_path(common["weight_dir"])
    result_dir = project_path(common["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Device: {device}")
    selected_case = str(common["case"]).lower()
    if selected_case not in (*CASES, "all"):
        raise ValueError("common.case must be one of: 1d, 2d, all")
    selected_cases = CASES if selected_case == "all" else (selected_case,)
    if "1d" in selected_cases:
        infer_poisson1d(weight_dir, result_dir, device, dtype)
    if "2d" in selected_cases:
        infer_poisson2d(weight_dir, result_dir, device, dtype)


if __name__ == "__main__":
    main()
