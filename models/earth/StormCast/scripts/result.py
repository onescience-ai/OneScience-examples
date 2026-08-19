from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize StormCast HDF5 forecasts")
    parser.add_argument("--config", type=Path, default=Path("conf/config.yaml"))
    parser.add_argument("--input", type=Path, default="./outputs/inference/forecast.h5")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state-variable")
    parser.add_argument("--background-variable")
    parser.add_argument("--step", type=int, action="append")
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Plot model-space values instead of applying dataset statistics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    project_root = config_path.parent.parent
    data_root = Path(config["data"]["root_dir"])
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    output_dir = args.output_dir or Path(config["inference"]["output_dir"]) / "plots"
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    visualize(
        input_path=args.input,
        output_dir=output_dir,
        data_root=data_root,
        state_variable=args.state_variable
        or config["inference"]["plot_state_variable"],
        background_variable=args.background_variable
        or config["inference"]["plot_background_variable"],
        steps=args.step,
        denormalize=not args.normalized,
    )


def visualize(
    input_path: Path,
    output_dir: Path,
    data_root: Path,
    state_variable: str,
    background_variable: str,
    steps: list[int] | None = None,
    denormalize: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with h5py.File(input_path, "r") as handle:
        state_variables = _decode_strings(handle.attrs["state_variables"])
        background_variables = _decode_strings(handle.attrs["background_variables"])
        state_index = _variable_index(state_variables, state_variable, "state")
        background_index = _variable_index(
            background_variables, background_variable, "background"
        )
        selected_steps = steps or list(range(handle["prediction"].shape[0]))
        for step in selected_steps:
            if not 0 <= step < handle["prediction"].shape[0]:
                raise IndexError(f"Step {step} is outside the forecast range")

        source_normalized = bool(handle.attrs.get("normalized", False))
        stats_by_year: dict[
            int,
            tuple[
                tuple[np.ndarray, np.ndarray],
                tuple[np.ndarray, np.ndarray],
            ],
        ] = {}

        for step in selected_steps:
            prediction = handle["prediction"][step, state_index]
            target = handle["target"][step, state_index]
            background = handle["background"][step, background_index]
            time_index = int(handle["time_index"][step])
            if denormalize and source_normalized:
                year = int(str(time_index)[:4])
                if year not in stats_by_year:
                    stats_by_year[year] = (
                        _read_stats(
                            data_root / "hrrr" / "data" / f"{year}.h5",
                            state_variables,
                        ),
                        _read_stats(
                            data_root / "era5" / "data" / f"{year}.h5",
                            background_variables,
                        ),
                    )
                state_stats, background_stats = stats_by_year[year]
                prediction = _denormalize(prediction, state_stats, state_index)
                target = _denormalize(target, state_stats, state_index)
                background = _denormalize(
                    background, background_stats, background_index
                )
            output = output_dir / f"forecast_{step:03d}_{state_variable}.png"
            _save_four_panel(
                prediction,
                target,
                background,
                state_variable,
                background_variable,
                time_index,
                output,
                normalized=source_normalized and not denormalize,
            )
            outputs.append(output)
            print(f"plot={output}")
    return outputs


def _save_four_panel(
    prediction: np.ndarray,
    target: np.ndarray,
    background: np.ndarray,
    state_variable: str,
    background_variable: str,
    time_index: int,
    output: Path,
    normalized: bool,
) -> None:
    error = prediction - target
    state_min = float(min(np.nanmin(prediction), np.nanmin(target)))
    state_max = float(max(np.nanmax(prediction), np.nanmax(target)))
    error_limit = max(float(np.nanmax(np.abs(error))), np.finfo(np.float32).eps)
    time_label = datetime.strptime(str(time_index), "%Y%m%d%H").strftime(
        "%Y-%m-%d %H:00"
    )
    units = " (normalized)" if normalized else ""
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    panels = (
        (
            prediction,
            f"StormCast {state_variable}{units}",
            "viridis",
            state_min,
            state_max,
        ),
        (target, f"Target {state_variable}{units}", "viridis", state_min, state_max),
        (background, f"ERA5 {background_variable}{units}", "magma", None, None),
        (
            error,
            f"Error {state_variable}{units}",
            "RdBu_r",
            -error_limit,
            error_limit,
        ),
    )
    for axis, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(
            data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(f"StormCast valid time: {time_label}", fontsize=13)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _read_stats(
    path: Path, expected_variables: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing statistics file: {path}")
    with h5py.File(path, "r") as handle:
        variables = _decode_strings(handle["fields"].attrs["variables"])
        if variables != expected_variables:
            raise ValueError(f"Variable order in {path} differs from inference output")
        means = np.asarray(handle["global_means"][:], dtype=np.float32).reshape(-1)
        stds = np.asarray(handle["global_stds"][:], dtype=np.float32).reshape(-1)
    return means, stds


def _denormalize(
    data: np.ndarray,
    stats: tuple[np.ndarray, np.ndarray],
    index: int,
) -> np.ndarray:
    means, stds = stats
    return data * stds[index] + means[index]


def _decode_strings(values: Any) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value) for value in values
    ]


def _variable_index(variables: list[str], name: str, kind: str) -> int:
    try:
        return variables.index(name)
    except ValueError as error:
        raise ValueError(f"Unknown {kind} variable {name!r}") from error


if __name__ == "__main__":
    main()
