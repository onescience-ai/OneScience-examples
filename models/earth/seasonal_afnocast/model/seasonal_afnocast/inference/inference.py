"""
Inference pipeline for seasonal post-processing with AFNOCast.

InferenceOrchestrator
---------------------
Runs end-to-end inference for a single SEAS5 forecast initialisation date
(given as "yyyymm").  The forecast spans 215 days and therefore crosses
7 or 8 calendar months of valid time.  Because one model state dict was
trained per target calendar month, the orchestrator:

    for each target month in the 215-day window
        1. load  trained_model/afnocast_best_<MM>.safetensors
        2. build InferenceDataset(forecast_file, target_month=M)
        3. run   batch inference  →  collect (predictions, metadata)

    stitch results into a single xarray Dataset and save to NetCDF.

Month routing is done in this script (not in the dataloader).
The InferenceDataset only produces samples for ONE target month at a time,
keeping the dataloader free of model-selection logic.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

from seasonal_afnocast.dataloader.dataloader_inference import (
    InferenceDataset,
    compute_target_months,
)
from seasonal_afnocast.models.AFNOCast import AFNOCast
from seasonal_afnocast.utils.config import load_config


# ---------------------------------------------------------------------------
# Default paths  (can be overridden via constructor arguments)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

_weights_dir = _PROJECT_ROOT / "weights"
_DEFAULT_TRAINED_MODEL_DIR = _weights_dir if _weights_dir.is_dir() else _PROJECT_ROOT
_DEFAULT_CONFIG_PATH       = _PROJECT_ROOT / "configs" / "config_afnocast.yaml"
_DEFAULT_OUTPUT_DIR        = _PROJECT_ROOT / "results" / "inference" 

# Output grid coordinates (ERA5-Land / CHIRPS 0.1° grid, Blue-Nile region)
# lat: 17.6 → 7.1 descending, 106 values
# lon: 32.0 → 39.7 ascending,  78 values
_LAT_HIGH = np.round(np.arange(17.6, 7.1, -0.1), 1).astype(np.float32)
_LON_HIGH = np.round(np.arange(32.0, 39.8,  0.1), 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class InferenceOrchestrator:
    """
    Run per-month inference over a full 215-day SEAS5 forecast.

    Parameters
    ----------
    init_date : str
        Initialisation date as ``"yyyymm"`` (e.g. ``"202509"``).
    config_path : str or Path, optional
        Path to the training config YAML that defines the model architecture.
        Defaults to ``configs/config_afnocast.yaml`` at the project root.
    trained_model_dir : str or Path, optional
        Directory containing ``afnocast_best_<MM>.safetensors`` state-dict files,
        one per calendar month.  Defaults to the project-root
        ``weights/`` folder.
    seas_forecast_dir : str or Path, optional
        Directory containing ``SEAS5_tp_<yyyymm>.nc`` forecast files.
    output_dir : str or Path, optional
        Directory where the output NetCDF will be written.
    device : str, optional
        PyTorch device string (e.g. ``"cuda:0"``).
        Defaults to ``"cuda"`` if available, else ``"cpu"``.
    batch_size : int, optional
        DataLoader batch size per inference step.
    num_workers : int, optional
        DataLoader worker count.
    input_processors : list of callables, optional
        Transforms passed to InferenceDataset (e.g. log_transform + Normalize).
        Should match exactly the processors used during training.
    coords_seas : list of float [lat_min, lat_max, lon_min, lon_max], optional
        Spatial crop for the SEAS5 input.  Defaults to the Blue-Nile region.
    """

    def __init__(
        self,
        init_date: str,
        config_path: Optional[str | pathlib.Path] = None,
        trained_model_dir: Optional[str | pathlib.Path] = None,
        seas_forecast_dir: Optional[str | pathlib.Path] = None,
        output_dir: Optional[str | pathlib.Path] = None,
        device: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        input_processors: Optional[list] = None,
        coords_seas: Optional[List[float]] = None,
    ) -> None:
        if len(init_date) != 6 or not init_date.isdigit():
            raise ValueError(
                f"init_date must be a 6-digit string 'yyyymm', got: {init_date!r}"
            )

        self.init_date  = init_date
        self.init_year  = init_date[:4]
        self.init_month = init_date[4:]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.input_processors = input_processors or []
        self.coords_seas      = coords_seas

        self.config = load_config(
            config_path or _DEFAULT_CONFIG_PATH
        )
        self.trained_model_dir = pathlib.Path(
            trained_model_dir or _DEFAULT_TRAINED_MODEL_DIR
        )
        if seas_forecast_dir is None:
            raise ValueError(
                "seas_forecast_dir must be provided: path to the directory "
                "containing SEAS5_tp_<yyyymm>.nc files."
            )
        self.seas_forecast_dir = str(seas_forecast_dir)

        self.output_dir = pathlib.Path(output_dir or _DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute the calendar months covered by the 215-day forecast window
        self.target_months: List[int] = compute_target_months(
            self.init_year, self.init_month
        )
        print(
            f"[InferenceOrchestrator] init_date={init_date} → "
            f"target months: {self.target_months}"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> xr.Dataset:
        """
        Run inference for every target month, assemble, save and return the
        combined xarray Dataset.

        Returns
        -------
        xr.Dataset
            Dimensions: ``(timestep, ens_out, lat_high, lon_high)``
        """
        all_preds: List[np.ndarray] = []
        all_metas: List[Dict]       = []

        for month in self.target_months:
            ckpt_path = self.trained_model_dir / f"afnocast_best_{month:02d}.safetensors"
            if not ckpt_path.exists():
                print(
                    f"  [month {month:02d}] State dict not found at {ckpt_path} — skipping."
                )
                continue

            print(f"  [month {month:02d}] Loading model from {ckpt_path}")
            model = self._load_model(ckpt_path)

            dataset = InferenceDataset(
                seas_forecast_dir=self.seas_forecast_dir,
                init_year=self.init_year,
                init_month=self.init_month,
                target_month=month,
                coords_seas=self.coords_seas,
                input_processors=self.input_processors,
            )

            if len(dataset) == 0:
                print(
                    f"  [month {month:02d}] No forecast days in this month — skipping."
                )
                continue

            print(
                f"  [month {month:02d}] Running inference on {len(dataset)} samples."
            )
            preds, metas = self._run_month(model, dataset)
            all_preds.extend(preds)
            all_metas.extend(metas)

            # Free VRAM between months
            del model
            torch.cuda.empty_cache()

        if not all_preds:
            raise RuntimeError(
                "Inference produced no output — check paths and target months."
            )

        ds_out = self._assemble(all_preds, all_metas)
        out_path = self.output_dir / f"SEAS5_AFNO_v1.0_tp_{self.init_date}.nc" # Outputfilename pattern according to SEAS5_BCSD_v3.0_tp_202505_0.1.nc
        ds_out.to_netcdf(out_path)
        print(f"[InferenceOrchestrator] Saved output → {out_path}")

        return ds_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_model_architecture(self) -> AFNOCast:
        """Instantiate the model from the config (architecture only, no weights)."""
        model_cfg = self.config.model
        model_type = model_cfg.get("type", "cnn")

        if model_type in ("cnn", "transformer", "conv_afno"):
            return AFNOCast(**model_cfg.params).to(self.device)

        raise ValueError(
            f"Unsupported model type for inference: {model_type!r}. "
            "Extend _init_model_architecture if needed."
        )

    def _load_model(self, ckpt_path: pathlib.Path) -> AFNOCast:
        """Load architecture, restore state dict, set eval mode."""
        from safetensors.torch import load_file

        model = self._init_model_architecture()
        state = load_file(ckpt_path, device=self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _run_month(
        self,
        model: AFNOCast,
        dataset: InferenceDataset,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Run inference for all samples of one target month.

        Returns
        -------
        preds : list of np.ndarray
            One array per sample, shape ``(ens_out, lat_high, lon_high)``.
        metas : list of dict
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

        preds_month: List[np.ndarray] = []
        metas_month: List[Dict]       = []

        with torch.no_grad():
            for x_sp, x_cond, metas_batch in loader:
                x_sp   = x_sp.to(self.device)                      # (B, ens_mem, 11, lat, lon)
                x_cond = x_cond.float().to(self.device)            # (B, 11, lat, lon)

                output = model(x_sp, x_cond)                       # (B, ens_out, [1,] lat_h, lon_h)

                # Squeeze any residual temporal singleton dim produced by the decoder
                if output.ndim == 5 and output.shape[2] == 1:
                    output = output.squeeze(2)                     # → (B, ens_out, lat_h, lon_h)

                output_np = output.cpu().numpy()                   # (B, ens_out, lat_h, lon_h)

                for i, meta in enumerate(metas_batch):
                    preds_month.append(output_np[i])               # (ens_out, lat_h, lon_h)
                    metas_month.append(meta)

        return preds_month, metas_month

    @staticmethod
    def _assemble(
        preds: List[np.ndarray],
        metas: List[Dict],
    ) -> xr.Dataset:
        """
        Stack per-sample predictions into a labelled xarray Dataset.

        Output dimensions: ``(timestep, ens, lat_high, lon_high)``
        """
        # Sort chronologically by valid time
        order = sorted(range(len(metas)), key=lambda i: metas[i]["timestep"])
        preds = [preds[i] for i in order]
        metas = [metas[i] for i in order]

        # Stack: (T, ens, lat_h, lon_h)
        arr = np.stack(preds, axis=0)

        timesteps   = [pd.Timestamp(m["timestep"])  for m in metas]
        doys        = [m["doy"]                     for m in metas]
        init_labels = [m["init_date_str"]           for m in metas]
        target_mons = [m["target_month"]            for m in metas]

        n_ens_out = arr.shape[1]
        lat_h     = arr.shape[2]
        lon_h     = arr.shape[3]

        # Validate output grid size matches known coordinate arrays
        if lat_h != len(_LAT_HIGH):
            raise ValueError(
                f"Model output lat dim ({lat_h}) does not match "
                f"expected grid size ({len(_LAT_HIGH)}). "
                "Update _LAT_HIGH in inference.py if the output grid has changed."
            )
        if lon_h != len(_LON_HIGH):
            raise ValueError(
                f"Model output lon dim ({lon_h}) does not match "
                f"expected grid size ({len(_LON_HIGH)}). "
                "Update _LON_HIGH in inference.py if the output grid has changed."
            )

        ds = xr.Dataset(
            data_vars={
                "tp": (
                    ["time", "ens", "lat", "lon"],
                    arr,
                    {"long_name": "Post-processed precipitation forecast",
                     "units": "mm/day"},
                ),
            },
            coords={
                "time": (
                    ["time"],
                    np.array(timesteps, dtype="datetime64[ns]"),
                ),
                "ens":   (["ens"],   np.arange(n_ens_out)),
                "lat":  (["lat"],  _LAT_HIGH,
                              {"units": "degrees_north", "long_name": "latitude"}),
                "lon": (["lon"], _LON_HIGH,
                              {"units": "degrees_east",  "long_name": "longitude"}),
                "init_date": init_labels[0],
            },
            attrs={
                "init_date":     init_labels[0],
                "description":   "Seasonal post-processing inference output (Seasonal AFNOCast) of SEAS5",
                "target_months": str(sorted(set(target_mons))),
                "version" : "3.0",
                "references": "Wiegels, R. et al. (2026): Improved Seasonal Precipitation Forecasts for the Blue Nile Basin: A Deep Learning Approach, doi: to be added", ### Update doi once available (https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2026.1691030/abstract)
                "institution" : "Karlsruhe Institute of Technology - Institute of Meteorology and Climate Research",
                "originator" : "Rebecca Wiegels",
                "contact" : "rebecca.wiegels@kit.edu",
                "contributor": "Christof Lorenz, Christian Chwala, Luca Glawion, Julius Polz, Tanja C. Schober, Harald Kunstmann",
                "source" : "ECMWF Seasonal Forecasts SEAS5",
                "license" : "CC BY-NC 4.0",
                "creation_date" : pd.Timestamp.now().isoformat(),
                "geospatial_lat_min" : float(_LAT_HIGH.min()),
                "geospatial_lat_max" : float(_LAT_HIGH.max()),
                "geospatial_lon_min" : float(_LON_HIGH.min()),
                "geospatial_lon_max" : float(_LON_HIGH.max()),
            },
        )

        return ds


# ---------------------------------------------------------------------------
# DataLoader collate helper
# ---------------------------------------------------------------------------

def _collate_fn(
    batch: List[Tuple[torch.Tensor, np.ndarray, Dict]]
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Stack x_sp and x_cond tensors; keep metas as a plain list of dicts.
    (torch's default_collate cannot handle dict values with mixed types.)
    """
    x_sps   = torch.stack([item[0] for item in batch], dim=0)
    x_conds = torch.tensor(
        np.stack([item[1] for item in batch], axis=0), dtype=torch.float32
    )
    metas = [item[2] for item in batch]
    return x_sps, x_conds, metas


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run seasonal post-processing inference for one SEAS5 forecast."
    )
    parser.add_argument(
        "--init_date",
        type=str,
        help="Initialisation date as 'yyyymm', e.g. 202509",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training_config.yaml (model architecture).",
    )
    parser.add_argument(
        "--trained-model-dir",
        type=str,
        default=None,
        help="Directory with afnocast_best_MM.safetensors files.",
    )
    parser.add_argument(
        "--seas-forecast-dir",
        type=str,
        default=None,
        help="Directory with SEAS5_tp_yyyymm.nc files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write output NetCDF.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device string, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--num-workers",  type=int, default=4)
    args = parser.parse_args()

    orchestrator = InferenceOrchestrator(
        init_date=args.init_date,
        config_path=args.config,
        trained_model_dir=args.trained_model_dir,
        seas_forecast_dir=args.seas_forecast_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    orchestrator.run()