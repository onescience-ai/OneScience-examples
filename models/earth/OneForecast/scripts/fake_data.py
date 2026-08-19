"""Generate metadata-compatible ERA5 HDF5 fixtures at the native 0.25 degree grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


OFFICIAL_VARIABLES = (
    [f"Z{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + [f"Q{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + [f"T{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + [f"U{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + [f"V{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + ["U10M", "V10M", "T2M", "MSLP"]
)

VARIABLE_ALIASES = {
    **{f"Z{x}": f"geopotential_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    **{f"Q{x}": f"specific_humidity_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    **{f"T{x}": f"temperature_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    **{f"U{x}": f"u_component_of_wind_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    **{f"V{x}": f"v_component_of_wind_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    "U10M": "10m_u_component_of_wind",
    "V10M": "10m_v_component_of_wind",
    "T2M": "2m_temperature",
    "MSLP": "mean_sea_level_pressure",
}
REAL_VARIABLES = tuple(VARIABLE_ALIASES[name] for name in OFFICIAL_VARIABLES)


def _synthetic_field(time_index: int, channel: int, height: int, width: int) -> np.ndarray:
    lat = np.linspace(1.0, -1.0, height, dtype=np.float32)[:, None]
    lon = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False, dtype=np.float32)[None, :]
    phase = np.float32(channel * 0.17)
    field = np.sin(lon + phase) + 0.4 * np.cos(np.float32(time_index / 3.0) + phase) + 0.2 * lat
    return np.asarray(field, dtype=np.float32)


def generate_fake_h5(output_dir: Path, years: list[int], stats_years: set[int], time_steps: int,
                     height: int, width: int, seed: int) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sums = np.zeros(len(OFFICIAL_VARIABLES), dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    value_count = 0

    for offset, year in enumerate(years):
        path = data_dir / f"{year}.h5"
        with h5py.File(path, "w") as handle:
            dataset = handle.create_dataset(
                "fields", shape=(time_steps, len(OFFICIAL_VARIABLES), height, width),
                dtype="float32", chunks=(1, 1, height, width),
            )
            dataset.attrs["variables"] = list(REAL_VARIABLES)
            dataset.attrs["time_step"] = 6
            for time_index in range(time_steps):
                for channel in range(len(OFFICIAL_VARIABLES)):
                    field = _synthetic_field(time_index + offset, channel, height, width)
                    dataset[time_index, channel] = field
                    if year in stats_years:
                        sums[channel] += field.sum(dtype=np.float64)
                        squared_sums[channel] += np.square(field, dtype=np.float64).sum()
            if year in stats_years:
                value_count += time_steps * height * width
        print(f"{path}: fields={(time_steps, len(OFFICIAL_VARIABLES), height, width)}, variables={len(OFFICIAL_VARIABLES)}")

    means = (sums / value_count).reshape(1, -1, 1, 1)
    variances = squared_sums / value_count - np.square(means.reshape(-1))
    stds = np.sqrt(np.maximum(variances, 1e-12)).reshape(1, -1, 1, 1)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "global_means.npy", means)
    np.save(stats_dir / "global_stds.npy", stds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument("--years", nargs="+", type=int, default=[2000, 2001, 2002])
    parser.add_argument("--stats-years", nargs="+", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--height", type=int, default=721)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.years) < 1 or min(args.time_steps, args.height, args.width) < 1:
        parser.error("years, time-steps, height, and width must be positive")
    if (args.height, args.width) != (721, 1440):
        parser.error("ERA5 fixtures must use the native 0.25 degree grid 721x1440")
    stats_years = set(args.stats_years or args.years[:1])
    if not stats_years.issubset(args.years):
        parser.error("stats-years must be included in years")
    generate_fake_h5(args.output_dir, args.years, stats_years, args.time_steps,
                     args.height, args.width, args.seed)


if __name__ == "__main__":
    main()
