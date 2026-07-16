from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    checkpoint_state,
    load_config,
    project_path,
    relative_l2,
    resolve_device,
    resolve_dtype,
)
from data_utils import batched_predict, load_allen_cahn  # noqa: E402
from model.fpinn import build_model  # noqa: E402


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
    weight_dir = project_path(common["weight_dir"], PROJECT_ROOT)
    result_dir = project_path(common["result_dir"], PROJECT_ROOT)
    checkpoint_path = weight_dir / task_config["output"]["checkpoint_name"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"invalid FPINN checkpoint: {checkpoint_path}")
    state, metadata = checkpoint_state(checkpoint)
    checkpoint_task = metadata.get("task", task)
    if checkpoint_task != task:
        raise ValueError(
            f"checkpoint task is {checkpoint_task}, but common.task is {task}"
        )
    model_config = metadata.get("model_config", config["model"])
    pde_config = metadata.get("pde_config", task_config["pde"])
    data_config = dict(metadata.get("data_config", config["data"]))
    data_path = project_path(data_config["mat_file"], PROJECT_ROOT)
    dataset = load_allen_cahn(data_path)
    model = build_model(task, model_config, pde_config, dtype).to(
        device=device, dtype=dtype
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    prediction = batched_predict(
        model,
        np.asarray(dataset["coordinates"]),
        int(data_config["evaluation_batch_size"]),
        device,
        dtype,
    )
    exact = np.asarray(dataset["exact"])
    error = relative_l2(prediction, exact)
    rmse = float(np.sqrt(np.mean((prediction - exact) ** 2)))
    max_error = float(np.max(np.abs(prediction - exact)))

    result_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = result_dir / task_config["output"]["predictions_name"]
    figure_path = result_dir / task_config["output"]["figure_name"]
    payload = {
        "coordinates": dataset["coordinates"],
        "exact": exact,
        "prediction": prediction,
        "relative_l2": error,
        "rmse": rmse,
        "max_abs_error": max_error,
    }
    if task == "inverse":
        payload["lambda_1"] = model.lambda_1.item()
        payload["lambda_2"] = model.lambda_2.item()
    np.savez_compressed(predictions_path, **payload)

    shape = dataset["grid_shape"]
    exact_grid = exact.reshape(shape)
    prediction_grid = prediction.reshape(shape)
    error_grid = np.abs(prediction_grid - exact_grid)
    extent = (
        float(np.min(dataset["space"])),
        float(np.max(dataset["space"])),
        float(np.min(dataset["time"])),
        float(np.max(dataset["time"])),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, title, field in zip(
        axes,
        ("Exact", "FPINN", "Absolute error"),
        (exact_grid, prediction_grid, error_grid),
        strict=True,
    ):
        image = axis.imshow(
            field,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="viridis" if title != "Absolute error" else "hot",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    title = f"Allen-Cahn {task} | L2={error:.3e}"
    if task == "inverse":
        title += (
            f" | lambda=({model.lambda_1.item():.3e}, "
            f"{model.lambda_2.item():.3e})"
        )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    print(f"Config: {config_path}")
    print(f"Task: {task}")
    print(f"Data: {data_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Relative L2={error:.6e}, RMSE={rmse:.6e}, MaxAbs={max_error:.6e}")
    if task == "inverse":
        print(
            f"Recovered lambda_1={model.lambda_1.item():.8e}, "
            f"lambda_2={model.lambda_2.item():.8e}"
        )
    print(f"Predictions: {predictions_path}")
    print(f"Plot: {figure_path}")


if __name__ == "__main__":
    main()
