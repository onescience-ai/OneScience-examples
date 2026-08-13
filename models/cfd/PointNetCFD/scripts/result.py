#!/usr/bin/env python3
"""Render PointCFD ground truth, prediction, and absolute field errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_string = str(PROJECT_ROOT)
if project_root_string in sys.path:
    sys.path.remove(project_root_string)
sys.path.insert(0, project_root_string)

from scripts.common import resolve_path, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "results" / "predictions.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--case-offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(PROJECT_ROOT, str(args.predictions))
    output_dir = resolve_path(PROJECT_ROOT, str(args.output_dir))
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction archive not found: {predictions_path}")
    if args.num_cases <= 0:
        raise ValueError("num-cases must be positive")
    if args.case_offset < 0:
        raise ValueError("case-offset cannot be negative")

    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {"coordinates", "predictions", "targets", "case_indices", "target_names"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Prediction archive is missing keys: {sorted(missing)}")
        coordinates = np.asarray(archive["coordinates"], dtype=np.float32)
        predictions = np.asarray(archive["predictions"], dtype=np.float32)
        targets = np.asarray(archive["targets"], dtype=np.float32)
        case_indices = np.asarray(archive["case_indices"], dtype=np.int64)
        target_names = [str(name) for name in archive["target_names"].tolist()]
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(f"coordinates must be [cases,points,2], got {coordinates.shape}")
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("predictions and targets must share [cases,points,variables]")
    if coordinates.shape[:2] != predictions.shape[:2]:
        raise ValueError("Coordinate and field case/point dimensions do not match")
    if predictions.shape[-1] != len(target_names):
        raise ValueError("target_names does not match prediction channels")
    if not (np.isfinite(coordinates).all() and np.isfinite(predictions).all() and np.isfinite(targets).all()):
        raise ValueError("Visualization inputs contain NaN or Infinity")

    stop = min(args.case_offset + args.num_cases, coordinates.shape[0])
    if args.case_offset >= stop:
        raise ValueError("case-offset is beyond the available predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []
    case_summaries: List[Dict[str, Any]] = []
    for local_index in range(args.case_offset, stop):
        xy = coordinates[local_index]
        case_prediction = predictions[local_index]
        case_target = targets[local_index]
        absolute_error = np.abs(case_prediction - case_target)
        rows = len(target_names)
        figure, axes = plt.subplots(rows, 3, figsize=(13.5, 4.1 * rows), squeeze=False)
        variable_summary: Dict[str, Any] = {}
        for channel, name in enumerate(target_names):
            lower = float(min(case_target[:, channel].min(), case_prediction[:, channel].min()))
            upper = float(max(case_target[:, channel].max(), case_prediction[:, channel].max()))
            if upper <= lower:
                upper = lower + 1.0e-12
            panels = (
                (case_target[:, channel], "Ground truth", lower, upper, "viridis"),
                (case_prediction[:, channel], "Prediction", lower, upper, "viridis"),
                (absolute_error[:, channel], "Absolute error", 0.0, None, "magma"),
            )
            for column, (values, title, vmin, vmax, color_map) in enumerate(panels):
                axis = axes[channel, column]
                scatter = axis.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    c=values,
                    s=8,
                    marker="o",
                    linewidths=0,
                    cmap=color_map,
                    vmin=vmin,
                    vmax=vmax,
                )
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.set_title(f"{name}: {title}")
                figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
            variable_summary[name] = {
                "mean_absolute_error": float(np.mean(absolute_error[:, channel])),
                "max_absolute_error": float(np.max(absolute_error[:, channel])),
            }
        case_index = int(case_indices[local_index])
        figure.suptitle(f"PointCFD fixed test case {case_index}")
        figure.tight_layout()
        output_path = output_dir / f"case_{case_index:04d}_fields.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        generated.append(str(output_path))
        case_summaries.append({"case_index": case_index, "variables": variable_summary})
        print(f"figure={output_path}", flush=True)

    summary = {
        "predictions": str(predictions_path),
        "visualization_method": (
            "direct point scatter without triangulation or interpolation because mesh topology "
            "and obstacle boundaries are not provided"
        ),
        "generated_files": generated,
        "cases": case_summaries,
    }
    summary_path = output_dir / "visualization_summary.json"
    write_json(summary_path, summary)
    print(f"visualization_summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
