from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from common import load_config

matplotlib.use("Agg")


def main():
    cfg = load_config()
    output_dir = Path(cfg.inference.output_dir)
    metrics_path = output_dir / "rollout_metrics.npz"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run scripts/inference.py before result.py."
        )

    mse = np.load(metrics_path)["mse"]
    print(f"Sequences: {len(mse)}")
    print(f"Average MSE: {float(np.mean(mse)):.6e}")
    print(f"Best MSE: {float(np.min(mse)):.6e}")
    print(f"Worst MSE: {float(np.max(mse)):.6e}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(mse)), mse, marker=".")
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Position MSE")
    ax.set_title("Lagrangian MeshGraphNet Rollout Error")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "error.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
