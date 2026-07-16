from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import torch
import yaml
from matplotlib.patches import Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import (  # noqa: E402
    build_evaluation_points,
    combined_coordinates,
    combined_exact_solution,
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


def load_model(
    checkpoint_path: Path,
    fallback_model_config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> XPINNPoisson2D:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"invalid XPINN checkpoint: {checkpoint_path}")
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
        model_config = checkpoint.get("model_config", fallback_model_config)
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state = checkpoint
        model_config = fallback_model_config
    else:
        raise ValueError(f"checkpoint contains no model state: {checkpoint_path}")
    model = XPINNPoisson2D(model_config, dtype=dtype).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict(
    model: XPINNPoisson2D, evaluation_points: dict[str, torch.Tensor]
) -> np.ndarray:
    with torch.no_grad():
        predictions = model.predict(
            evaluation_points["xy1"],
            evaluation_points["xy2"],
            evaluation_points["xy3"],
        )
    return torch.cat(predictions).cpu().numpy().reshape(-1)


def mask_polygon(data: Mapping) -> np.ndarray:
    boundary_x = np.asarray(data["xb"]).reshape(-1)
    boundary_y = np.asarray(data["yb"]).reshape(-1)
    return np.vstack(
        (
            np.column_stack((boundary_x, boundary_y)),
            np.array(
                [
                    [1.8, boundary_y[-1]],
                    [1.8, -1.7],
                    [-1.6, -1.7],
                    [-1.6, 1.55],
                    [1.8, 1.55],
                    [1.8, boundary_y[-1]],
                ]
            ),
            np.array([[boundary_x[-1], boundary_y[-1]]]),
        )
    )


def save_figure(
    data: Mapping,
    exact: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    x, y = combined_coordinates(data)
    if exact.size != x.size or prediction.size != x.size:
        raise ValueError("plot coordinates, exact values, and predictions must have equal size")
    triangulation = tri.Triangulation(x, y)
    polygon = mask_polygon(data)
    fields = (
        ("Exact", exact),
        ("XPINN", prediction),
        ("Absolute error", np.abs(exact - prediction)),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, (title, field) in zip(axes, fields, strict=True):
        contour = axis.tricontourf(triangulation, field, 100, cmap="jet")
        axis.add_patch(
            Polygon(polygon, closed=True, facecolor="white", edgecolor="white")
        )
        axis.plot(data["xi1"].reshape(-1), data["yi1"].reshape(-1), "w-", linewidth=0.5)
        axis.plot(data["xi2"].reshape(-1), data["yi2"].reshape(-1), "w-", linewidth=0.5)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        figure.colorbar(contour, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    common = config["common"]
    device = resolve_device(str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    data_path = project_path(config["data"]["mat_file"])
    weight_dir = project_path(common["weight_dir"])
    result_dir = project_path(common["result_dir"])
    checkpoint_path = weight_dir / config["training"]["checkpoint_name"]
    output_path = result_dir / config["inference"]["figure_name"]
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Data: {data_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    data = load_mat_data(data_path)
    model = load_model(checkpoint_path, config["model"], device, dtype)
    evaluation_points = build_evaluation_points(data, device, dtype)
    prediction = predict(model, evaluation_points)
    exact = combined_exact_solution(data)
    relative_l2 = float(
        np.linalg.norm(exact - prediction) / np.linalg.norm(exact)
    )
    save_figure(data, exact, prediction, output_path)
    print(f"Relative L2={relative_l2:.6e}")
    print(f"Plot: {output_path}")


if __name__ == "__main__":
    main()
