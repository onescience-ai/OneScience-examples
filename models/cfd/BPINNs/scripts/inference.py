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
    build_laplace_data,
    checkpoint_state,
    load_config,
    project_path,
    relative_l2,
    resolve_device,
    resolve_dtype,
)
from model.bpinn import build_model, posterior_predict  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    common = config["common"]
    device = resolve_device(str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    weight_dir = project_path(common["weight_dir"], PROJECT_ROOT)
    result_dir = project_path(common["result_dir"], PROJECT_ROOT)
    refined_path = weight_dir / config["training"]["refined_checkpoint_name"]
    base_path = weight_dir / config["training"]["checkpoint_name"]
    checkpoint_path = refined_path if refined_path.is_file() else base_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"invalid BPINN checkpoint: {checkpoint_path}")
    state, metadata = checkpoint_state(checkpoint)
    model_config = metadata.get("model_config", config["model"])
    data_config = dict(metadata.get("data_config", config["data"]))
    seed = int(metadata.get("seed", common["seed"]))
    data = build_laplace_data(data_config, seed, device, dtype)
    model = build_model(model_config, dtype=dtype).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.eval()

    posterior_states = metadata.get("posterior_states") or [state]
    mean, standard_deviation, samples = posterior_predict(
        model, posterior_states, data["x_test"]
    )
    mean_numpy = mean.cpu().numpy()
    std_numpy = standard_deviation.cpu().numpy()
    exact_numpy = data["u_test"].cpu().numpy()
    x_numpy = data["x_test"].cpu().numpy()
    error = relative_l2(mean_numpy, exact_numpy)
    rmse = float(np.sqrt(np.mean((mean_numpy - exact_numpy) ** 2)))
    max_error = float(np.max(np.abs(mean_numpy - exact_numpy)))

    result_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = result_dir / config["inference"]["predictions_name"]
    figure_path = result_dir / config["inference"]["figure_name"]
    np.savez_compressed(
        predictions_path,
        x=x_numpy,
        exact=exact_numpy,
        mean=mean_numpy,
        std=std_numpy,
        samples=samples.cpu().numpy(),
        relative_l2=error,
        rmse=rmse,
        max_abs_error=max_error,
    )

    x_axis = x_numpy.reshape(-1)
    exact_axis = exact_numpy.reshape(-1)
    mean_axis = mean_numpy.reshape(-1)
    std_axis = std_numpy.reshape(-1)
    absolute_error = np.abs(mean_axis - exact_axis)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x_axis, exact_axis, "k-", linewidth=1.5, label="Exact")
    axes[0].plot(x_axis, mean_axis, "r--", linewidth=1.5, label="BPINN")
    if np.any(std_axis > 0):
        axes[0].fill_between(
            x_axis,
            mean_axis - 2.0 * std_axis,
            mean_axis + 2.0 * std_axis,
            color="red",
            alpha=0.2,
            label="2 std",
        )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend()
    axes[1].semilogy(x_axis, np.maximum(absolute_error, 1.0e-16), "b-")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute error")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Posterior states: {len(posterior_states)}")
    print(f"Relative L2={error:.6e}, RMSE={rmse:.6e}, MaxAbs={max_error:.6e}")
    print(f"Predictions: {predictions_path}")
    print(f"Plot: {figure_path}")


if __name__ == "__main__":
    main()
