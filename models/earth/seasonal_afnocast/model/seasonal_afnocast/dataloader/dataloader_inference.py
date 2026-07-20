"""
Inference dataloader for seasonal post-processing.

Reads a single raw SEAS5 forecast NetCDF file
  /bg/data/s2s/TABN/01_raw_forecasts/initial_resolution/SEAS5_tp_yyyymm.nc
and prepares per-day samples with ±5-day temporal context windows (11 steps
total), filtered to a single target calendar month (month of valid time).

No ERA5 / CHIRPS target is loaded — this is for inference only.

The sample tensor shapes produced by __getitem__ are identical to those of
SeasonalDataset (dataloader_zarr.py), so existing model forward passes and
transforms work without modification:

    x_sp   : torch.Tensor  (ens_mem, time_window=11, lat, lon)  — SEAS5 input
    x_cond : np.ndarray    (time_window=11, lat, lon)            — ensemble std
    meta   : dict          timestep, doy, init_date_str, target_month

The ±5-day padding at forecast boundaries replicates the first / last available
day, exactly as in data_generator_all() (dataloader_utils.py).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """
    PyTorch Dataset for inference on a single raw SEAS5 forecast file.

    Parameters
    ----------
    seas_forecast_dir : str
        Directory that contains the SEAS5 forecast files.
        Expected filename pattern: ``SEAS5_tp_<yyyymm>.nc``
        (e.g. ``/bg/data/s2s/TABN/01_raw_forecasts/initial_resolution/``)
    init_year : str
        4-digit initialisation year, e.g. ``"2025"``.
    init_month : str
        Zero-padded initialisation month, e.g. ``"09"``.
    target_month : int
        Calendar month of valid time to process (1–12).
        Only forecast days whose valid date falls in this month are included.
    coords_seas : list of float, optional
        Bounding box [lat_min, lat_max, lon_min, lon_max] for spatial crop.
        Defaults to the Blue-Nile region used in training.
    input_processors : list of callables, optional
        Transforms applied to ``x_sp`` and ``x_cond``
        (e.g. ``log_transform``, ``Normalize(mean, std)``).
        Applied in list order.
    """

    DEFAULT_COORDS_SEAS: List[float] = [7.0, 17.8, 32.0, 39.7]

    def __init__(
        self,
        seas_forecast_dir: str,
        init_year: str,
        init_month: str,
        target_month: int,
        coords_seas: Optional[List[float]] = None,
        input_processors: Optional[List[Callable]] = None,
    ) -> None:
        self.seas_forecast_dir = seas_forecast_dir
        self.init_year = init_year
        self.init_month = init_month
        self.target_month = target_month
        self.coords_seas = coords_seas if coords_seas is not None else self.DEFAULT_COORDS_SEAS
        self.input_processors = input_processors or []

        # List of (windowed_ds, valid_timestamp, init_label)
        self._samples: List[Tuple[xr.Dataset, np.datetime64, str]] = []
        self._prepare_samples()

    # ------------------------------------------------------------------
    # Sample preparation  (adapted from data_generator_all in dataloader_utils.py)
    # ------------------------------------------------------------------

    def _prepare_samples(self) -> None:
        """
        Load the SEAS5 file, clip spatially, filter to *target_month* of
        valid time, and build an 11-step ±5-day context window for each day.
        """
        fcst_path = os.path.join(
            self.seas_forecast_dir,
            f"SEAS5_tp_{self.init_year}{self.init_month}.nc",
        )
        if not os.path.exists(fcst_path):
            raise FileNotFoundError(
                f"SEAS5 forecast file not found: {fcst_path}"
            )

        # Open and normalise dtypes (same as data_generator_all)
        ds_seas = xr.open_dataset(fcst_path, chunks={})
        if "ens" in ds_seas.dims:
            ds_seas = ds_seas.rename({"ens": "ens_mem"})

        ds_seas["tp"] = ds_seas.tp.astype("float32")
        ds_seas["lat"] = ds_seas.lat.astype("float32")
        ds_seas["lon"] = ds_seas.lon.astype("float32")

        # Spatial crop
        cs = self.coords_seas
        ds_seas = ds_seas.where(
            np.logical_and(ds_seas.lat >= cs[0], ds_seas.lat <= cs[1]),
            drop=True,
        )
        ds_seas = ds_seas.where(
            np.logical_and(ds_seas.lon >= cs[2], ds_seas.lon <= cs[3]),
            drop=True,
        )

        # Filter valid-time days to the requested target month
        ds_month = ds_seas.where(
            ds_seas.time.dt.month == self.target_month, drop=True
        )
        if ds_month.time.size == 0:
            # No forecast days fall in this calendar month — dataset is empty
            return

        init_label = f"{self.init_year}-{self.init_month}"

        for ts in ds_month.time.values:
            ts_m5 = ts - np.timedelta64(5, "D")
            ts_p5 = ts + np.timedelta64(5, "D")

            # Distance from first / last available forecast day
            d_first = int(
                (ts - ds_seas.isel(time=0).time.values)
                / np.timedelta64(1, "D")
            )
            d_last = int(
                (ds_seas.isel(time=-1).time.values - ts)
                / np.timedelta64(1, "D")
            )

            ds_sub = _build_window(ds_seas, ts, ts_m5, ts_p5, d_first, d_last)
            self._samples.append((ds_sub, ts, init_label))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        ds_sub, ts, init_label = self._samples[idx]

        # Shape after _build_window: (ens_mem, time_window=11, lat, lon)
        x: np.ndarray = ds_sub.tp.values

        x_sp = torch.tensor(x, dtype=torch.float32)

        # Ensemble std per pixel over the temporal window — used as conditioning
        x_cond: np.ndarray = np.std(x, axis=0)  # (time_window, lat, lon)

        # Apply transforms (same processors used during training)
        x_sp = _apply_processors(x_sp, self.input_processors)
        x_cond = _apply_processors(x_cond, self.input_processors)

        meta = {
            "timestep": str(pd.Timestamp(ts)),
            "doy": int(pd.Timestamp(ts).timetuple().tm_yday),
            "init_date_str": init_label,
            "target_month": self.target_month,
        }

        return x_sp, x_cond, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_window(
    ds_full: xr.Dataset,
    ts: np.datetime64,
    ts_m5: np.datetime64,
    ts_p5: np.datetime64,
    d_first: int,
    d_last: int,
) -> xr.Dataset:
    """
    Build an 11-step (±5-day) temporal context window around *ts*.

    When fewer than 5 days are available at the start / end of the forecast,
    the first / last available day is repeated to pad the window — identical
    to the padding strategy in data_generator_all() (dataloader_utils.py).

    Returns a Dataset with dims ``(ens_mem, time, lat, lon)`` where
    ``time`` has exactly 11 steps.
    """
    if d_first <= 4:
        ds_sub_mini = ds_full.sel(time=slice(ts_m5, ts_p5))
        ds_edge = ds_full.isel(time=0)
        n_pad = 5 - d_first
        ds_sub = xr.concat([ds_edge] * n_pad + [ds_sub_mini], dim="time")
    elif d_last <= 4:
        ds_sub_mini = ds_full.sel(time=slice(ts_m5, ts_p5))
        ds_edge = ds_full.isel(time=-1)
        n_pad = 5 - d_last
        ds_sub = xr.concat([ds_sub_mini] + [ds_edge] * n_pad, dim="time")
    else:
        ds_sub = ds_full.sel(time=slice(ts_m5, ts_p5))

    return ds_sub.transpose("ens_mem", "time", "lat", "lon")


def _apply_processors(data, processors: List[Callable]):
    """Apply a list of transforms in sequence."""
    for processor in processors:
        if processor is not None:
            data = processor(data)
    return data


def compute_target_months(init_year: str, init_month: str) -> List[int]:
    """
    Return the sorted list of calendar months (1–12) covered by a 215-day
    SEAS5 forecast issued on the first of *init_month*.

    ``InferenceOrchestrator`` uses this to know which per-month state dicts
    need to be loaded.

    Parameters
    ----------
    init_year : str
        4-digit year string (e.g. ``"2025"``).
    init_month : str
        Zero-padded month string (e.g. ``"09"``).

    Returns
    -------
    list of int
        Sorted unique calendar months present in the 215-day window.
        Typically 7 or 8 months depending on the initialisation date.

    Example
    -------
    >>> compute_target_months("2025", "09")
    [9, 10, 11, 12, 1, 2, 3, 4]   # 2025-09-01 to 2026-04-04
    """
    init_date = pd.Timestamp(f"{init_year}-{init_month}-01")
    end_date = init_date + pd.Timedelta(days=214)  # 215 days inclusive
    months = pd.date_range(start=init_date, end=end_date, freq="D").month.unique().tolist()
    return sorted(months)
