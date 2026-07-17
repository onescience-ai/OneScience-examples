from __future__ import annotations

import argparse

import numpy as np

from common import load_config, project_path, relative_errors, write_json


def plot_result(prediction: np.ndarray, target: np.ndarray, cfg: dict, figure_path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skip figure: {exc}")
        return False

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x_min, x_max = cfg["data"]["x_range"]
    t_min, t_max = cfg["data"]["t_range"]
    items = [
        (prediction, "Predicted u(x,t)"),
        (target, "Exact u(x,t)"),
        (np.abs(prediction - target), "Absolute Error"),
    ]
    for ax, (data, title) in zip(axes, items):
        image = ax.imshow(data, extent=[x_min, x_max, t_max, t_min], aspect="auto")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and plot PINNsformer inference output.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--prediction", default=None, help="Path to prediction .npz.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prediction_path = project_path(args.prediction or cfg["paths"]["prediction"])
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}. Run scripts/inference.py first.")

    data = np.load(prediction_path)
    prediction = data["prediction"]
    target = data["target"]
    metrics = relative_errors(prediction, target)

    metrics_path = project_path(cfg["paths"]["metrics"])
    write_json(metrics_path, metrics)
    figure_path = project_path(cfg["paths"]["figure"])
    plotted = plot_result(prediction, target, cfg, figure_path)

    print(f"metrics saved to {metrics_path}")
    if plotted:
        print(f"figure saved to {figure_path}")
    print(f"relative L1 error: {metrics['relative_l1']:.6f}")
    print(f"relative L2 error: {metrics['relative_l2']:.6f}")


if __name__ == "__main__":
    main()
