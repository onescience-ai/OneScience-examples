#!/usr/bin/env python3
"""Visualize CNO fields and ID/OOD relative-L1 distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="array index to plot; default is the sample nearest the split median error",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction artifact not found: {path}")
    with np.load(path) as payload:
        required = {"sample_ids", "inputs", "targets", "predictions", "relative_l1"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        arrays = {key: np.asarray(payload[key]) for key in required}
    count = arrays["sample_ids"].shape[0]
    for key in ("inputs", "targets", "predictions", "relative_l1"):
        if arrays[key].shape[0] != count:
            raise ValueError(f"sample count mismatch for {key} in {path}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"nonfinite values in {key} from {path}")
    if arrays["inputs"].ndim != 4 or arrays["inputs"].shape[1] != 1:
        raise ValueError(f"expected N1HW fields in {path}, got {arrays['inputs'].shape}")
    return arrays


def representative_index(errors: np.ndarray, requested: int | None) -> int:
    if requested is not None:
        if requested < 0 or requested >= errors.size:
            raise IndexError(f"sample-index {requested} outside [0,{errors.size})")
        return requested
    median = np.median(errors)
    return int(np.argmin(np.abs(errors - median)))


def plot_fields(
    split: str,
    arrays: dict[str, np.ndarray],
    output_path: Path,
    requested_index: int | None,
    dpi: int,
) -> None:
    errors_percent = arrays["relative_l1"] * 100.0
    index = representative_index(errors_percent, requested_index)
    input_field = arrays["inputs"][index, 0]
    target = arrays["targets"][index, 0]
    prediction = arrays["predictions"][index, 0]
    absolute_error = np.abs(prediction - target)
    field_min = float(min(target.min(), prediction.min()))
    field_max = float(max(target.max(), prediction.max()))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    input_image = axes[0].imshow(input_field, origin="lower", cmap="RdBu_r")
    fig.colorbar(input_image, ax=axes[0], shrink=0.78)
    target_image = axes[1].imshow(
        target, origin="lower", cmap="RdBu_r", vmin=field_min, vmax=field_max
    )
    prediction_image = axes[2].imshow(
        prediction, origin="lower", cmap="RdBu_r", vmin=field_min, vmax=field_max
    )
    error_image = axes[3].imshow(absolute_error, origin="lower", cmap="magma")
    fig.colorbar(target_image, ax=[axes[1], axes[2]], shrink=0.78)
    fig.colorbar(error_image, ax=axes[3], shrink=0.78)
    titles = ("Initial velocity", "Target at T=1", "CNO prediction", "Absolute error")
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
    sample_id = int(arrays["sample_ids"][index])
    fig.suptitle(
        f"{split.upper()} Sample_{sample_id} — relative L1={errors_percent[index]:.3f}%"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_distribution(
    split_arrays: dict[str, dict[str, np.ndarray]],
    paper_reference: dict,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"id": "#2166ac", "ood": "#b2182b"}
    for split in ("id", "ood"):
        values = split_arrays[split]["relative_l1"] * 100.0
        axes[0].hist(values, bins=20, alpha=0.55, label=split.upper(), color=colors[split])
        axes[0].axvline(
            np.median(values), color=colors[split], linewidth=2, linestyle="-"
        )
        reference = paper_reference.get(split)
        if reference is not None:
            axes[0].axvline(
                float(reference), color=colors[split], linewidth=1.5, linestyle="--"
            )
    axes[0].set_title("Per-sample relative L1")
    axes[0].set_xlabel("Relative L1 (%)")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="solid=reproduction\ndashed=paper")

    values = [
        split_arrays["id"]["relative_l1"] * 100.0,
        split_arrays["ood"]["relative_l1"] * 100.0,
    ]
    box = axes[1].boxplot(values, tick_labels=["ID", "OOD"], patch_artist=True)
    for patch, color in zip(box["boxes"], (colors["id"], colors["ood"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axes[1].set_title("Error distribution summary")
    axes[1].set_ylabel("Relative L1 (%)")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    split_arrays = {
        split: load_predictions(results_dir / f"{split}_predictions.npz")
        for split in ("id", "ood")
    }
    for split, arrays in split_arrays.items():
        output_path = results_dir / f"{split}_fields.png"
        plot_fields(split, arrays, output_path, args.sample_index, args.dpi)
        values = arrays["relative_l1"] * 100.0
        print(
            f"visualization split={split} median={np.median(values):.6f}% "
            f"mean={np.mean(values):.6f}% saved={output_path}",
            flush=True,
        )

    distribution_path = results_dir / "error_distribution.png"
    plot_distribution(
        split_arrays,
        metrics.get("paper_reference", {}),
        distribution_path,
        args.dpi,
    )
    print(f"visualization saved={distribution_path}", flush=True)


if __name__ == "__main__":
    main()
