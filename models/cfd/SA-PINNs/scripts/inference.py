from __future__ import annotations

import argparse
import os
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
    resolve_device,
    resolve_dtype,
)
from model.sa_pinn import build_model  # noqa: E402
from problems import CASES, generate_data, point_counts, relative_l2, to_tensors  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SA-PINNs inference and visualization")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("SAPINN_CONFIG", DEFAULT_CONFIG)),
        help="YAML configuration file",
    )
    parser.add_argument("--case", choices=CASES, default="laplace")
    parser.add_argument("--device", help="Override common.device")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path")
    parser.add_argument("--weight-dir", type=Path, help="Override weight directory")
    parser.add_argument("--result-dir", type=Path, help="Override result directory")
    parser.add_argument("--test-res", type=int, help="Override test resolution per axis")
    return parser.parse_args()


def save_figure(
    case: str,
    data: Mapping,
    prediction: np.ndarray,
    attention: np.ndarray | None,
    output_path: Path,
    error: float | None,
) -> None:
    if case == "laplace":
        exact = data["u_exact"].reshape(-1)
        predicted = prediction.reshape(-1)
        x_axis = data["x_test"].reshape(-1)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(x_axis, exact, "k-", label="Exact")
        axes[0].plot(x_axis, predicted, "r--", label="SA-PINN")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u")
        axes[0].legend()
        axes[1].semilogy(
            x_axis, np.maximum(np.abs(predicted - exact), 1.0e-16), "r-"
        )
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("Absolute error")
        if attention is not None:
            order = np.argsort(data["x_pde"].reshape(-1))
            axes[2].plot(
                data["x_pde"].reshape(-1)[order],
                attention.reshape(-1)[order],
                "b-",
            )
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("Attention weight")
        else:
            axes[2].text(0.5, 0.5, "Attention disabled", ha="center", va="center")
            axes[2].set_axis_off()
    elif case == "helmholtz":
        shape = data["test_shape"]
        exact = data["u_exact"].reshape(shape)
        predicted = prediction.reshape(shape)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, title, field in zip(
            axes,
            ("Exact", "SA-PINN", "Absolute error"),
            (exact, predicted, np.abs(predicted - exact)),
            strict=True,
        ):
            image = axis.imshow(field, extent=(-1, 1, -1, 1), origin="lower", cmap="jet")
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            figure.colorbar(image, ax=axis)
    else:
        shape = data["test_shape"]
        predicted = prediction.reshape(shape)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        image = axes[0].imshow(
            predicted,
            extent=(-1, 1, 0, 1),
            origin="lower",
            aspect="auto",
            cmap="jet",
        )
        axes[0].set_title("SA-PINN Burgers prediction")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("t")
        figure.colorbar(image, ax=axes[0])
        if attention is not None:
            scatter = axes[1].scatter(
                data["x_pde"][:, 0],
                data["x_pde"][:, 1],
                c=attention.reshape(-1),
                s=8,
                cmap="viridis",
            )
            axes[1].set_title("PDE attention weights")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("t")
            figure.colorbar(scatter, ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, "Attention disabled", ha="center", va="center")
            axes[1].set_axis_off()
    figure.suptitle(f"{case} | L2={'N/A' if error is None else f'{error:.3e}'}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    common = config["common"]
    case_config = config["cases"][args.case]
    device = resolve_device(args.device or str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    weight_dir = project_path(args.weight_dir or common["weight_dir"], PROJECT_ROOT)
    result_dir = project_path(args.result_dir or common["result_dir"], PROJECT_ROOT)
    checkpoint_path = (
        project_path(args.checkpoint, PROJECT_ROOT)
        if args.checkpoint
        else weight_dir / case_config["output"]["checkpoint_name"]
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. Run scripts/train.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"invalid SA-PINN checkpoint: {checkpoint_path}")
    state, metadata = checkpoint_state(checkpoint)
    checkpoint_case = metadata.get("case", args.case)
    if checkpoint_case != args.case:
        raise ValueError(
            f"checkpoint case is {checkpoint_case}, but --case requested {args.case}"
        )
    model_config = metadata.get("model_config", case_config["model"])
    data_config = dict(metadata.get("data_config", case_config["data"]))
    if args.test_res is not None:
        data_config["test_res"] = args.test_res
    seed = int(metadata.get("seed", common["seed"]))
    stored_counts = metadata.get("point_counts")
    if stored_counts is None:
        stored_counts = {
            "pde": int(state["att_pde.alpha"].shape[0])
            if "att_pde.alpha" in state
            else int(data_config["n_pde"]),
            "data": int(state["att_data.alpha"].shape[0])
            if "att_data.alpha" in state
            else int(data_config["n_sol"]),
            "boundary": int(state["att_boundary.alpha"].shape[0])
            if "att_boundary.alpha" in state
            else int(data_config["n_bnd"]),
        }
        data_config.update(
            {
                "n_pde": stored_counts["pde"],
                "n_sol": stored_counts["data"],
                "n_bnd": stored_counts["boundary"],
            }
        )
    data = generate_data(args.case, data_config, seed)
    counts = stored_counts or point_counts(data)
    attention_enabled = bool(
        metadata.get(
            "attention_enabled", any(key.startswith("att_") for key in state)
        )
    )
    model = build_model(model_config, counts, attention_enabled, dtype).to(
        device=device, dtype=dtype
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    tensors = to_tensors(data, device, dtype)
    with torch.no_grad():
        prediction = model(tensors["x_test"]).cpu().numpy()
    error = relative_l2(prediction, data["u_exact"])
    attention = (
        model.att_pde().detach().cpu().numpy() if model.att_pde is not None else None
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = result_dir / case_config["output"]["predictions_name"]
    figure_path = result_dir / case_config["output"]["figure_name"]
    np.savez_compressed(
        predictions_path,
        coordinates=data["x_test"],
        prediction=prediction,
        exact=np.array([]) if data["u_exact"] is None else data["u_exact"],
        relative_l2=np.nan if error is None else error,
        pde_attention=np.array([]) if attention is None else attention,
    )
    save_figure(args.case, data, prediction, attention, figure_path, error)

    print(f"Config: {config_path}")
    print(f"Case: {args.case}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Relative L2={'N/A' if error is None else f'{error:.6e}'}")
    if attention is not None:
        print(
            f"PDE attention: mean={attention.mean():.6f}, "
            f"std={attention.std():.6f}, min={attention.min():.6f}, "
            f"max={attention.max():.6f}"
        )
    print(f"Predictions: {predictions_path}")
    print(f"Plot: {figure_path}")


if __name__ == "__main__":
    main()
