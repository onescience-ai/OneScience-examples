import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize GP_for_TO inference outputs.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def save_plot(data, output_dir):
    x = data["x"]
    fields = [("u", data["u"]), ("v", data["v"]), ("p", data["p"]), ("ro", data["ro"])]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (name, values) in zip(axes.reshape(-1), fields):
        im = ax.tricontourf(x[:, 0], x[:, 1], values, levels=32, cmap="viridis")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = output_dir / "field_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    cfg = load_config()
    output_dir = resolve_path(args.output_dir or cfg["inference"]["output_dir"])
    pred_path = output_dir / "predictions.npz"
    summary_path = output_dir / "inference_summary.json"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing inference output: {pred_path}")

    data = np.load(pred_path)
    print(f"Prediction file: {pred_path}")
    for name in ("x", "u", "v", "p", "ro"):
        arr = data[name]
        print(
            f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={float(arr.min()):.6e}, max={float(arr.max()):.6e}, mean={float(arr.mean()):.6e}"
        )

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"Problem: {summary.get('problem')}, checkpoint source: {summary.get('checkpoint_metadata', {}).get('problem')}")

    if not args.no_plot:
        plot_path = save_plot(data, output_dir)
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
