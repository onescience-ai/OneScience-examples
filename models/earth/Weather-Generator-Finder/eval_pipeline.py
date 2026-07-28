#!/usr/bin/env python3
"""
================================================================================
AI Weather Model Evaluation Pipeline
================================================================================
Phase 1: Data preparation (FREE — runs in CPU sandbox)
Phase 2+: GPU inference on HF jobs (requires budget)

Usage:
  python eval_pipeline.py --phase data_prep    # Downloads WeatherBench2 subset
  python eval_pipeline.py --phase eval --model aurora --region Western_Europe
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# ---------------------------------------------------------------------------
# REGION DEFINITIONS (lat_min, lat_max, lon_min, lon_max)
# ---------------------------------------------------------------------------
REGIONS = {
    "North_America":        (15,  75, -170, -50),
    "South_America":        (-55,  15,  -90, -30),
    "Western_Europe":       (35,  70,  -15,  30),
    "Eastern_Europe":       (40,  70,   30,  60),
    "North_Africa":         (15,  38,  -20,  40),
    "Central_South_Africa": (-35, 15,   10,  55),
    "Middle_East":          (12,  42,   30,  60),
    "South_Asia":           (5,   38,   60,  95),
    "East_Asia":            (18,  55,   95, 145),
    "Southeast_Asia":       (-11, 28,   95, 145),
    "Oceania":              (-50, -5,  110, 180),
    "Arctic":               (60,  90, -180, 180),
    "Antarctica":           (-90, -60, -180, 180),
    "Tropical_Pacific":     (-20,  20,  120, -90),
}

# ---------------------------------------------------------------------------
# MODEL CONFIGS — HF Hub repos, framework, estimated inference time per IC
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "aurora": {
        "hub_repo": "microsoft/aurora",
        "framework": "pytorch",
        "min_per_ic_6h": 5,      # minutes per initial condition on A10G
        "min_per_ic_3d": 8,
        "variables": ["t2m", "tp"],
        "resolution": 0.25,
        "needs_install": ["microsoft/aurora", "torch"],
    },
    "aifs_single_1_0": {
        "hub_repo": "ecmwf/aifs-single-1.0",
        "framework": "anemoi",   # ECMWF's framework
        "min_per_ic_6h": 8,
        "min_per_ic_3d": 12,
        "variables": ["t2m", "tp"],
        "resolution": 0.25,
        "needs_install": ["anemoi-models", "torch"],
    },
    "graphcast_era5_37L": {
        "hub_repo": "shermansiu/dm_graphcast",
        "framework": "jax",
        "min_per_ic_6h": 10,
        "min_per_ic_3d": 15,
        "variables": ["t2m", "tp"],
        "resolution": 0.25,
        "needs_install": ["dm-haiku", "jax[cuda12_pip]", "graphcast @ git+https://github.com/google-deepmind/graphcast.git"],
    },
    "graphcast_amse": {
        "hub_repo": "csubich/graphcast_amse",
        "framework": "jax",
        "min_per_ic_6h": 10,
        "min_per_ic_3d": 15,
        "variables": ["t2m", "tp"],
        "resolution": 0.25,
        "needs_install": ["dm-haiku", "jax[cuda12_pip]"],
    },
    "pangu_weather_1h": {
        "hub_repo": "xiaobai10086/pangu_weather_1.onnx",
        "framework": "onnx",
        "min_per_ic_6h": 3,
        "min_per_ic_3d": 5,   # cascaded 1h steps
        "variables": ["t2m", "tp"],
        "resolution": 0.25,
        "needs_install": ["onnxruntime-gpu", "numpy"],
    },
    "weathernext2_gencast": {
        "hub_repo": "openclimatefix/gencast-128x64",
        "framework": "pytorch",
        "min_per_ic_6h": 15,
        "min_per_ic_3d": 20,
        "variables": ["t2m", "tp"],
        "resolution": 0.25,
        "needs_install": ["torch", "diffusers"],
    },
}

# ---------------------------------------------------------------------------
# PHASE 1: DATA PREPARATION
# ---------------------------------------------------------------------------
def phase_data_prep(
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-07",
    variables: List[str] = None,
    data_dir: str = "./weatherbench_data",
    mock: bool = False,
):
    """
    Download WeatherBench2 ERA5 test data subset and create regional masks.
    This runs on CPU — no GPU needed.
    
    If mock=True, generates synthetic data with correct structure for pipeline testing.
    """
    if variables is None:
        variables = ["2m_temperature", "total_precipitation_6hr"]

    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 80)
    print(f"\nDate range: {start_date} to {end_date}")
    print(f"Variables: {variables}")
    print(f"Output: {out.resolve()}")
    print(f"Mock mode: {mock}")

    local_path = out / "era5_2020_6h.zarr"

    if mock:
        print(f"\n[MOCK MODE] Generating synthetic ERA5-like dataset...")
        return _generate_mock_data(start_date, end_date, variables, out, local_path)

    # WeatherBench2 Zarr URL
    wb2_url = "gs://weatherbench2/datasets/era5/2020-2021_6h_1440x721.zarr"

    print(f"\n[1/4] Checking WeatherBench2 data access...")
    print(f"      Source: {wb2_url}")
    print(f"      Local:  {local_path}")

    try:
        import xarray as xr
        print("      ✓ xarray available")
    except ImportError:
        print("      ✗ xarray NOT installed. Run: pip install xarray zarr gcsfs")
        return False

    try:
        import zarr
        print("      ✓ zarr available")
    except ImportError:
        print("      ✗ zarr NOT installed")
        return False

    try:
        import gcsfs
        print("      ✓ gcsfs available")
    except ImportError:
        print("      ⚠ gcsfs NOT installed.")
        return False

    print(f"\n[2/4] Loading WeatherBench2 dataset (lazy — metadata only)...")
    try:
        ds = xr.open_zarr(wb2_url, consolidated=True)
        print(f"      ✓ Dataset loaded")
        print(f"      Dimensions: {dict(ds.dims)}")
        print(f"      Available vars: {list(ds.data_vars)[:10]}...")
    except Exception as e:
        print(f"      ✗ Failed to load: {e}")
        print(f"\n      GCS access requires Google Cloud authentication.")
        print(f"      Options:")
        print(f"        1. Authenticate: gcloud auth application-default login")
        print(f"        2. Download data via alternative source (see weatherbench2.ai)")
        print(f"        3. Use mock mode for pipeline testing:")
        print(f"           python eval_pipeline.py --phase data_prep --mock")
        return False

    print(f"\n[3/4] Subsetting to date range and variables...")
    try:
        ds_subset = ds.sel(time=slice(start_date, end_date))
        available = set(ds_subset.data_vars)
        wanted = set(variables)
        found = available & wanted
        missing = wanted - available
        if missing:
            print(f"      ⚠ Missing variables: {missing}")
            print(f"      Available: {list(available)[:20]}...")
        ds_subset = ds_subset[list(found)]
        print(f"      ✓ Subset: {dict(ds_subset.dims)}")
    except Exception as e:
        print(f"      ✗ Subset failed: {e}")
        return False

    print(f"\n[4/4] Saving subset to local Zarr...")
    try:
        ds_subset.to_zarr(local_path, mode="w", consolidated=True)
        print(f"      ✓ Saved to {local_path}")
        size_mb = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file()) / 1e6
        print(f"      Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"      ✗ Save failed: {e}")
        return False

    return _finalize_prep(out, local_path, wb2_url, start_date, end_date, list(found), list(missing))


def _generate_mock_data(start_date, end_date, variables, out, local_path):
    """Generate synthetic ERA5-like data for pipeline testing."""
    import xarray as xr
    import pandas as pd
    import numpy as np

    # ERA5-like grid: 721 lat × 1440 lon, 0.25° resolution
    lats = np.linspace(-90, 90, 721)
    lons = np.linspace(0, 359.75, 1440)
    times = pd.date_range(start=start_date, end=end_date, freq="6h")

    print(f"      Grid: {len(lats)} lat × {len(lons)} lon")
    print(f"      Times: {len(times)} steps")

    data_vars = {}
    for var in variables:
        # Generate synthetic data with realistic patterns
        np.random.seed(42)
        # T2M: mean ~288K, seasonal + latitudinal gradient
        if "temperature" in var.lower() or var == "t2m":
            base = 288 - 30 * np.cos(np.radians(lats[:, None]))
            noise = np.random.randn(len(times), len(lats), len(lons)) * 2
            data = base[None, :, :] + noise
            data_vars[var] = (["time", "latitude", "longitude"], data.astype(np.float32))
        else:
            # Precip: sparse, skewed
            base = np.random.exponential(0.5, (len(times), len(lats), len(lons)))
            data_vars[var] = (["time", "latitude", "longitude"], base.astype(np.float32))

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        }
    )

    print(f"      Saving to Zarr...")
    ds.to_zarr(local_path, mode="w", consolidated=True)
    size_mb = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file()) / 1e6
    print(f"      ✓ Saved: {size_mb:.1f} MB")

    return _finalize_prep(out, local_path, "MOCK", start_date, end_date, variables, [])


def _finalize_prep(out, local_path, source, start_date, end_date, found, missing):
    """Create masks and metadata."""
    import xarray as xr

    print(f"\n[5/5] Creating regional masks...")
    ds_subset = xr.open_zarr(local_path, consolidated=True)
    masks = {}
    lats = ds_subset.latitude.values
    lons = ds_subset.longitude.values
    for region, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        if lon_min < lon_max:
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
        else:
            lon_mask = (lons >= lon_min) | (lons <= lon_max)
        mask = lat_mask[:, None] & lon_mask[None, :]
        masks[region] = mask.astype(np.uint8)

    mask_path = out / "regional_masks.npz"
    np.savez_compressed(mask_path, **masks, latitude=lats, longitude=lons)
    print(f"      ✓ Masks saved to {mask_path}")

    meta = {
        "source": source,
        "date_range": [start_date, end_date],
        "variables": found,
        "missing_variables": missing,
        "regions": list(REGIONS.keys()),
        "local_path": str(local_path),
        "mask_path": str(mask_path),
        "resolution": 0.25,
        "mock": source == "MOCK",
    }
    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"      ✓ Metadata saved to {meta_path}")

    print(f"\n{'=' * 80}")
    print("PHASE 1 COMPLETE")
    print(f"{'=' * 80}")
    if source == "MOCK":
        print(f"\n⚠ This is MOCK data for pipeline testing only.")
        print(f"   For real rankings, download actual WeatherBench2 ERA5 data.")
    print(f"\nNext steps:")
    print(f"  1. Verify data: python eval_pipeline.py --phase verify")
    print(f"  2. Run evaluation (requires GPU jobs):")
    print(f"     python eval_pipeline.py --phase eval --model aurora --region Western_Europe")
    return True


# ---------------------------------------------------------------------------
# VERIFY
# ---------------------------------------------------------------------------
def phase_verify(data_dir: str = "./weatherbench_data"):
    """Verify downloaded data and masks are correct."""
    out = Path(data_dir)
    meta_path = out / "metadata.json"
    mask_path = out / "regional_masks.npz"
    data_path = out / "era5_2020_6h.zarr"

    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    if not meta_path.exists():
        print(f"✗ Metadata not found: {meta_path}")
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"✓ Metadata: {meta['date_range']}, {len(meta['variables'])} variables")

    if not mask_path.exists():
        print(f"✗ Masks not found: {mask_path}")
        return False
    masks = np.load(mask_path)
    print(f"✓ Masks: {len([k for k in masks.files if k not in ('latitude', 'longitude')])} regions")

    if not data_path.exists():
        print(f"✗ Data not found: {data_path}")
        return False
    try:
        import xarray as xr
        ds = xr.open_zarr(data_path, consolidated=True)
        print(f"✓ Data: {dict(ds.dims)}, {list(ds.data_vars)}")
    except Exception as e:
        print(f"✗ Data load failed: {e}")
        return False

    print("\n✓ All checks passed. Ready for evaluation.")
    return True


# ---------------------------------------------------------------------------
# COST ESTIMATOR
# ---------------------------------------------------------------------------
def estimate_cost(
    models: List[str],
    regions: List[str],
    leads: List[str],
    num_ic: int = 28,  # 1 week at 6h
    hardware: str = "a10g-large"
):
    """Estimate GPU job cost for a benchmark run."""
    pricing = {
        "t4-small": 0.60,
        "t4-medium": 0.90,
        "a10g-small": 2.00,
        "a10g-large": 2.00,
        "a100-large": 4.00,
    }
    price = pricing.get(hardware, 2.00)

    total_hours = 0
    breakdown = []
    for model in models:
        cfg = MODEL_CONFIGS.get(model)
        if not cfg:
            continue
        for lead in leads:
            mins_per_ic = cfg.get(f"min_per_ic_{lead}", 10)
            hours = num_ic * mins_per_ic / 60
            total_hours += hours
            breakdown.append({
                "model": model,
                "lead": lead,
                "ic": num_ic,
                "hours": round(hours, 1),
                "cost": round(hours * price, 2),
            })

    total_cost = total_hours * price

    print("=" * 80)
    print("COST ESTIMATE")
    print("=" * 80)
    print(f"\nModels:     {models}")
    print(f"Regions:    {regions} (computed from global, no extra cost)")
    print(f"Lead times:  {leads}")
    print(f"IC count:    {num_ic} ({num_ic * 6 / 4:.0f} days)")
    print(f"Hardware:    {hardware} (${price}/hr)")
    print(f"\n{'Model':<25} | {'Lead':<6} | {'IC':>4} | {'Hours':>7} | {'Cost':>7}")
    print("-" * 70)
    for b in breakdown:
        print(f"{b['model']:<25} | {b['lead']:<6} | {b['ic']:>4} | {b['hours']:>7.1f} | ${b['cost']:>6.2f}")
    print("-" * 70)
    print(f"{'TOTAL':<25} | {'':<6} | {'':>4} | {total_hours:>7.1f} | ${total_cost:>6.2f}")

    print(f"\nRECOMMENDED JOB COMMAND:")
    print(f"  hf_jobs run \\")
    print(f"    --script eval_{models[0]}.py \\")
    print(f"    --hardware {hardware} \\")
    print(f"    --timeout {int(total_hours / len(models)) + 2}h \\")
    print(f"    --dependencies {','.join(MODEL_CONFIGS[models[0]]['needs_install'])}")

    return total_cost


# ---------------------------------------------------------------------------
# EVALUATION (Skeleton — requires GPU + model-specific code)
# ---------------------------------------------------------------------------
def phase_eval(model: str, region: str, lead: str = "6h", data_dir: str = "./weatherbench_data"):
    """
    Run evaluation for a single model/region/lead.
    This is a SKELETON — actual implementation requires model-specific loading.
    """
    cfg = MODEL_CONFIGS.get(model)
    if not cfg:
        print(f"Unknown model: {model}")
        return False

    print("=" * 80)
    print(f"EVALUATION: {model} on {region} at {lead}")
    print("=" * 80)
    print(f"\nFramework: {cfg['framework']}")
    print(f"Hub repo:  {cfg['hub_repo']}")
    print(f"Variables: {cfg['variables']}")

    print(f"\n[!] This is a SKELETON.")
    print(f"    Actual implementation requires:")
    print(f"    1. Loading model weights from {cfg['hub_repo']}")
    print(f"    2. Running inference on GPU")
    print(f"    3. Slicing predictions to region mask")
    print(f"    4. Computing RMSE/ACC/ETS vs ERA5 truth")
    print(f"\n    To implement, add model-specific inference code in:")
    print(f"    - eval_aurora.py       (PyTorch)")
    print(f"    - eval_aifs.py         (anemoi)")
    print(f"    - eval_graphcast.py    (JAX)")
    print(f"    - eval_pangu.py        (ONNX)")
    print(f"    - eval_gencast.py      (PyTorch/diffusers)")

    print(f"\n    Example Aurora eval script:")
    print(f"""
    from aurora import AuroraModel, Batch, Metadata
    import torch
    import xarray as xr
    import numpy as np

    # Load model
    model = AuroraModel.from_pretrained("{cfg['hub_repo']}")
    model = model.cuda().eval()

    # Load ERA5 IC
    ds = xr.open_zarr("{data_dir}/era5_2020_6h.zarr")
    ic = ds.isel(time=0)  # first initial condition

    # Run inference
    with torch.no_grad():
        pred = model(ic)  # model-specific preprocessing needed

    # Slice to region
    mask = np.load("{data_dir}/regional_masks.npz")["{region}"]
    pred_region = pred.where(mask)

    # Compute RMSE
    truth = ds.isel(time=1)  # next timestep
    rmse = np.sqrt(((pred_region - truth)**2).mean())
    print(f"RMSE: {{rmse}}")
    """)

    return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI Weather Model Evaluation Pipeline")
    parser.add_argument("--phase", choices=["data_prep", "verify", "cost", "eval"], required=True)
    parser.add_argument("--model", default="aurora", help="Model to evaluate")
    parser.add_argument("--region", default="Western_Europe", help="Region name")
    parser.add_argument("--lead", default="6h", choices=["6h", "1d", "3d", "5d", "10d"])
    parser.add_argument("--data-dir", default="./weatherbench_data")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2020-01-07")
    parser.add_argument("--hardware", default="a10g-large")
    parser.add_argument("--models", nargs="+", default=["aurora", "aifs_single_1_0", "graphcast_era5_37L"])
    parser.add_argument("--regions", nargs="+", default=["Western_Europe", "North_America", "Tropical_Pacific"])
    parser.add_argument("--leads", nargs="+", default=["6h", "3d"])
    parser.add_argument("--mock", action="store_true", help="Use synthetic data for pipeline testing (no GCS download)")
    args = parser.parse_args()

    if args.phase == "data_prep":
        phase_data_prep(args.start_date, args.end_date, data_dir=args.data_dir, mock=args.mock)
    elif args.phase == "verify":
        phase_verify(args.data_dir)
    elif args.phase == "cost":
        estimate_cost(args.models, args.regions, args.leads, hardware=args.hardware)
    elif args.phase == "eval":
        phase_eval(args.model, args.region, args.lead, args.data_dir)


if __name__ == "__main__":
    main()
