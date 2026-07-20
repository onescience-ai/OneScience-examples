#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Cell 1: Environment setup (in Jupyter Notebook)
import sys
import os
from pathlib import Path

print("=" * 60)
print("Starting environment setup...")
print("=" * 60)

# Get current notebook directory
notebook_dir = Path(os.getcwd())
print(f"📁 Current working directory: {notebook_dir}")

# Determine project root
if notebook_dir.name == "scripts" and notebook_dir.parent.name == "model":
    project_root = notebook_dir.parent.parent
    print(f"📍 Detected: Running in model/scripts")
elif notebook_dir.name == "model":
    project_root = notebook_dir.parent
    print(f"📍 Detected: Running in model directory")
else:
    project_root = notebook_dir
    print(f"📍 Detected: Running in project root")

print(f"📁 Project root: {project_root}")

# Set paths - CORRECTED for your structure
model_outer_path = project_root / "model"                    # /.../model
model_inner_path = model_outer_path / "model"                # /.../model/model
seasonal_afnocast_path = model_inner_path / "seasonal_afnocast"  # /.../model/model/seasonal_afnocast
scripts_path = model_outer_path / "scripts"                  # /.../model/scripts

print(f"📁 Outer model path: {model_outer_path}")
print(f"📁 Inner model path: {model_inner_path}")
print(f"📁 seasonal_afnocast path: {seasonal_afnocast_path}")
print(f"📁 Scripts path: {scripts_path}")

# Check if directories exist
print("\n🔍 Checking if directories exist:")
print(f"   - Outer model: {model_outer_path.exists()}")
print(f"   - Inner model: {model_inner_path.exists()}")
print(f"   - seasonal_afnocast: {seasonal_afnocast_path.exists()}")
print(f"   - Scripts: {scripts_path.exists()}")

# If seasonal_afnocast exists, show its contents
if seasonal_afnocast_path.exists():
    print(f"\n📁 seasonal_afnocast contents:")
    for item in seasonal_afnocast_path.iterdir():
        print(f"      - {item.name}")
        # If it's a directory, show sub-contents
        if item.is_dir() and item.name in ["inference", "models", "dataloader", "utils"]:
            print(f"         📁 {item.name} contents:")
            for sub_item in item.iterdir():
                print(f"            - {sub_item.name}")

# Add paths to sys.path
paths_to_add = [
    str(seasonal_afnocast_path),   # Contains inference, models, dataloader, utils
    str(model_inner_path),         # Contains seasonal_afnocast
    str(model_outer_path),         # Contains config, weight, scripts
    str(scripts_path)              # Scripts directory
]

print("\n🔧 Adding paths to sys.path:")
for path in paths_to_add:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    print(f"   ✅ Added: {path}")

# Change to project root
os.chdir(project_root)
print(f"\n📂 Switched to project root: {os.getcwd()}")

print(f"\n✅ Environment setup complete!")
print(f"✅ Python path (first 4):")
for i, p in enumerate(sys.path[:4], 1):
    print(f"   {i}. {p}")

# Verify import
print("\n🔍 Attempting to import InferenceOrchestrator...")

# Try 1: Direct import from inference (if seasonal_afnocast_path is in sys.path)
try:
    from inference.inference import InferenceOrchestrator
    print("✅ Successfully imported from inference.inference")
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"   ❌ Direct import failed: {e}")
    IMPORT_SUCCESS = False
    
    # Try 2: Import with seasonal_afnocast prefix
    try:
        from seasonal_afnocast.inference.inference import InferenceOrchestrator
        print("✅ Successfully imported from seasonal_afnocast.inference.inference")
        IMPORT_SUCCESS = True
    except ImportError as e2:
        print(f"   ❌ seasonal_afnocast import failed: {e2}")
        IMPORT_SUCCESS = False

# If import failed, show debug info
if not IMPORT_SUCCESS:
    print(f"\n📁 Debug information:")
    print(f"   seasonal_afnocast_path: {seasonal_afnocast_path}")
    print(f"   seasonal_afnocast_path exists: {seasonal_afnocast_path.exists()}")
    
    if seasonal_afnocast_path.exists():
        print(f"\n   Contents of seasonal_afnocast:")
        for item in seasonal_afnocast_path.iterdir():
            print(f"      - {item.name}")
            if item.name == "inference" and item.is_dir():
                print(f"         Contents of inference:")
                for sub_item in item.iterdir():
                    print(f"            - {sub_item.name}")
    
    # Check if inference.py exists
    inference_file = seasonal_afnocast_path / "inference" / "inference.py"
    print(f"\n   inference.py exists: {inference_file.exists()}")
    if inference_file.exists():
        print(f"   ✅ inference.py found at: {inference_file}")
    else:
        # Check alternative location
        alt_inference = model_inner_path / "inference" / "inference.py"
        print(f"   Alternative inference.py exists: {alt_inference.exists()}")
        if alt_inference.exists():
            print(f"   ✅ inference.py found at: {alt_inference}")
    
    print(f"\n   sys.path entries:")
    for i, p in enumerate(sys.path[:5], 1):
        print(f"      {i}. {p}")

print("\n" + "=" * 60)
print("Environment setup complete!")
print("=" * 60)


# In[42]:


# Cell 2: Generate properly scaled mock SEAS5 data (model expects ~0.5)
import sys
import os
import numpy as np
import xarray as xr
import shutil
from pathlib import Path

print("=" * 60)
print("Generating mock SEAS5 data...")
print("=" * 60)

# Determine paths manually (robust method)
notebook_dir = Path(os.getcwd())

# Determine project root
if notebook_dir.name == "scripts" and notebook_dir.parent.name == "model":
    project_root = notebook_dir.parent.parent
    print(f"📍 Detected: Running in model/scripts")
elif notebook_dir.name == "model":
    project_root = notebook_dir.parent
    print(f"📍 Detected: Running in model directory")
else:
    project_root = notebook_dir
    print(f"📍 Detected: Running in project root")

print(f"📁 Project root: {project_root}")

# Set paths according to your structure
model_outer_path = project_root / "model"
model_inner_path = model_outer_path / "model"
seasonal_afnocast_path = model_inner_path / "seasonal_afnocast"
scripts_path = model_outer_path / "scripts"

print(f"📁 Model outer path: {model_outer_path}")
print(f"📁 Model inner path: {model_inner_path}")
print(f"📁 seasonal_afnocast path: {seasonal_afnocast_path}")
print(f"📁 Scripts path: {scripts_path}")

# ✅ Change: Save mock data to scripts directory instead of project root
mock_data_dir = scripts_path / "mock_data"  # /model/scripts/mock_data

# Add necessary paths if not already added
paths_to_add = [
    str(seasonal_afnocast_path),  # Contains inference, models, dataloader, utils
    str(model_inner_path),         # Contains seasonal_afnocast
    str(model_outer_path),         # Contains config, weight, scripts
    str(scripts_path)              # Scripts directory
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"   ✅ Added to sys.path: {path}")

# ✅ Don't change directory - stay in scripts
# os.chdir(project_root)  # Comment out or remove this line
print(f"\n📂 Staying in current directory: {os.getcwd()}")
print(f"📁 Mock data will be saved to: {mock_data_dir}")

def create_proper_mock_data(init_date="202505", output_dir=None):
    """
    Generate properly scaled mock SEAS5 data
    Model expects input scale around ~0.5
    """
    if output_dir is None:
        output_dir = str(mock_data_dir)
    
    print(f"\n🔄 Generating properly scaled mock data ({init_date})...")
    print(f"   Target scale: ~0.5 (model expects input around 0.5)")
    print(f"   Output directory: {output_dir}")
    
    # Grid definition (0.201°)
    lat = np.arange(7.1, 17.6, 0.201)[:38]
    lon = np.arange(32.0, 39.7, 0.201)[:28]
    
    # Time: target day ± 5 days = 11 days
    start_date = np.datetime64(f"{init_date[:4]}-{init_date[4:6]}-15")
    times = start_date + np.arange(-5, 6)
    ens_mem = np.arange(1, 26)  # 25 ensemble members
    
    np.random.seed(42)
    n_times, n_ens, n_lat, n_lon = len(times), len(ens_mem), len(lat), len(lon)
    
    # Spatial pattern
    lat_grid, lon_grid = np.meshgrid(
        np.linspace(0, 1, n_lat),
        np.linspace(0, 1, n_lon),
        indexing='ij'
    )
    spatial = 0.5 + 0.5 * np.exp(-((lat_grid-0.5)**2 + (lon_grid-0.6)**2) * 4)
    
    # ⚠️ KEY FIX: Scale data to ~0.5 (what model expects)
    target_scale = 0.5
    
    # Generate precipitation data
    data = np.zeros((n_times, n_ens, n_lat, n_lon))
    for t in range(n_times):
        time_factor = 1 + 0.3 * np.sin(t / 11 * np.pi)
        for e in range(n_ens):
            noise = 0.1 * np.random.randn(n_lat, n_lon)
            # Data range: 0 ~ target_scale * 1.3 ≈ 0.65
            data[t, e] = np.maximum(spatial * time_factor * target_scale + noise * 0.05, 0)
    
    # Save to NetCDF
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ds = xr.Dataset({
        "tp": xr.DataArray(
            data,
            dims=["time", "ens_mem", "lat", "lon"],
            coords={
                "time": times,
                "ens_mem": ens_mem,
                "lat": lat,
                "lon": lon
            },
            attrs={
                "units": "normalized",
                "long_name": "Total precipitation (normalized)",
                "note": "Scale ~0.5 matches model expectation"
            }
        )
    })
    
    file_path = output_path / f"SEAS5_tp_{init_date}.nc"
    ds.to_netcdf(file_path)
    
    print(f"✅ Data generated: {file_path}")
    print(f"   Dimensions: {dict(ds.sizes)}")
    print(f"   Data mean: {data.mean():.4f}")
    print(f"   Data max: {data.max():.4f}")
    print(f"   Data min: {data.min():.4f}")
    print(f"   ✅ Scale matches model expectation (~0.5)")
    return file_path

# Clean and generate properly scaled data
if mock_data_dir.exists():
    shutil.rmtree(mock_data_dir)
    print("\n🧹 Cleaned old data")

data_path = create_proper_mock_data("202505")

# Quick verification
print("\n📋 Verifying data:")
ds_check = xr.open_dataset(data_path)
print(f"   Dimensions: {dict(ds_check.sizes)}")
print(f"   tp range: [{ds_check.tp.min().values:.4f}, {ds_check.tp.max().values:.4f}]")
print(f"   tp mean: {ds_check.tp.mean().values:.4f}")
ds_check.close()

print("\n" + "=" * 60)
print("✅ Mock data generation complete!")
print(f"   Data saved to: {data_path}")
print("=" * 60)


# In[43]:


# Cell 3: Verify all required files
print("=" * 60)
print("📦 File verification")
print("=" * 60)

# Get project root
from pathlib import Path
import os

# Determine project root (same logic as Cell 1)
notebook_dir = Path(os.getcwd())
if notebook_dir.name == "scripts" and notebook_dir.parent.name == "model":
    project_root = notebook_dir.parent.parent
elif notebook_dir.name == "model":
    project_root = notebook_dir.parent
else:
    project_root = notebook_dir

print(f"📁 Project root: {project_root}")

# Check data files (mock_data in project root)
data_dir = project_root / "mock_data"
if data_dir.exists():
    nc_files = list(data_dir.glob("*.nc"))
    print(f"✅ Data files: {len(nc_files)} found")
    for f in sorted(nc_files):
        print(f"   - {f.name}")
else:
    print(f"❌ Data directory not found: {data_dir}")

# Check weight files (weight directory inside model outer path)
weights_dir = project_root / "model" / "weight"
if weights_dir.exists():
    # Check for .safetensors files
    weight_files = list(weights_dir.glob("*.safetensors"))
    if weight_files:
        print(f"\n✅ Weight files: {len(weight_files)} found")
        # Show first 5 and last 5 to avoid too much output
        for f in sorted(weight_files)[:5]:
            size = f.stat().st_size / (1024*1024)
            print(f"   - {f.name} ({size:.2f} MB)")
        if len(weight_files) > 10:
            print(f"   ... and {len(weight_files) - 10} more files")
            for f in sorted(weight_files)[-5:]:
                size = f.stat().st_size / (1024*1024)
                print(f"   - {f.name} ({size:.2f} MB)")
    else:
        # Check for other weight formats
        other_weights = list(weights_dir.glob("*.pth")) + list(weights_dir.glob("*.pt")) + list(weights_dir.glob("*.ckpt"))
        if other_weights:
            print(f"\n✅ Weight files: {len(other_weights)} found")
            for f in sorted(other_weights):
                size = f.stat().st_size / (1024*1024)
                print(f"   - {f.name} ({size:.2f} MB)")
        else:
            print(f"\n⚠️  No weight files found in: {weights_dir}")
            print(f"   Directory contents:")
            for item in weights_dir.iterdir():
                print(f"      - {item.name}")
else:
    print(f"\n❌ Weight directory not found: {weights_dir}")

# Check config file (config directory inside model outer path)
config_file = project_root / "model" / "config" / "config_afnocast.yaml"
if config_file.exists():
    print(f"\n✅ Config file: {config_file}")
else:
    print(f"\n⚠️  Config file not found: {config_file}")
    # Check if there are other config files
    config_dir = project_root / "model" / "config"
    if config_dir.exists():
        print(f"\n   Config directory contents:")
        for item in config_dir.iterdir():
            print(f"      - {item.name}")

# Check if model directory structure is correct
print("\n📁 Model directory structure check:")

# Check outer model
model_outer_path = project_root / "model"
print(f"   Model outer path: {model_outer_path}")
print(f"   ✅ Exists: {model_outer_path.exists()}")

# Check inner model
model_inner_path = model_outer_path / "model"
print(f"\n   Model inner path: {model_inner_path}")
print(f"   ✅ Exists: {model_inner_path.exists()}")

if model_inner_path.exists():
    print(f"\n   📁 Contents of model/model/:")
    for item in model_inner_path.iterdir():
        print(f"      - {item.name}")
    
    # Check seasonal_afnocast
    seasonal_afnocast_path = model_inner_path / "seasonal_afnocast"
    print(f"\n   📁 seasonal_afnocast path: {seasonal_afnocast_path}")
    print(f"   ✅ Exists: {seasonal_afnocast_path.exists()}")
    
    if seasonal_afnocast_path.exists():
        print(f"\n   📁 Contents of seasonal_afnocast/:")
        expected_dirs = ["inference", "models", "dataloader", "utils"]
        found_dirs = []
        missing_dirs = []
        
        for item in seasonal_afnocast_path.iterdir():
            print(f"      - {item.name}")
            if item.is_dir():
                found_dirs.append(item.name)
        
        # Check expected directories
        print(f"\n   🔍 Checking expected directories:")
        for dir_name in expected_dirs:
            dir_path = seasonal_afnocast_path / dir_name
            if dir_path.exists():
                print(f"      ✅ {dir_name}/ exists")
                # Show contents of each directory
                print(f"         Contents:")
                for sub_item in dir_path.iterdir():
                    print(f"            - {sub_item.name}")
            else:
                print(f"      ❌ {dir_name}/ missing")
                missing_dirs.append(dir_name)
        
        # Summary
        if missing_dirs:
            print(f"\n   ⚠️  Missing directories: {', '.join(missing_dirs)}")
            print(f"   💡 Expected structure: model/model/seasonal_afnocast/{{inference, models, dataloader, utils}}/")
        else:
            print(f"\n   ✅ All expected directories found!")
    else:
        print(f"\n   ❌ seasonal_afnocast directory not found!")
        print(f"   💡 Expected: {seasonal_afnocast_path}")

print("\n" + "=" * 60)
print("File verification complete!")
print("=" * 60)


# In[44]:


# Cell 4: Run inference
import torch
import xarray as xr
import sys
import os
from pathlib import Path

print("=" * 60)
print("🚀 Starting inference")
print("=" * 60)

# Determine project root
notebook_dir = Path(os.getcwd())
if notebook_dir.name == "scripts" and notebook_dir.parent.name == "model":
    project_root = notebook_dir.parent.parent
elif notebook_dir.name == "model":
    project_root = notebook_dir.parent
else:
    project_root = notebook_dir

print(f"📁 Project root: {project_root}")

# Set paths
model_outer_path = project_root / "model"
model_inner_path = model_outer_path / "model"
seasonal_afnocast_path = model_inner_path / "seasonal_afnocast"
scripts_path = model_outer_path / "scripts"

# Add paths to sys.path
paths_to_add = [
    str(seasonal_afnocast_path),
    str(model_inner_path),
    str(model_outer_path),
    str(scripts_path)
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n💻 Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ✅ FIX: Read data from scripts/mock_data (where Cell 2 saved it)
data_file = scripts_path / "mock_data" / "SEAS5_tp_202505.nc"
print(f"\n📁 Data file: {data_file}")
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}")

# Verify data format
ds_verify = xr.open_dataset(data_file)
print(f"\n📊 Data verification:")
print(f"   Dimensions: {dict(ds_verify.sizes)}")
print(f"   Expected: {{'time': 11, 'ens_mem': 25, 'lat': 38, 'lon': 28}}")
if set(ds_verify.dims.keys()) == {'time', 'ens_mem', 'lat', 'lon'}:
    print("   ✅ Dimensions match!")
else:
    print(f"   ❌ Dimensions mismatch! Actual: {set(ds_verify.dims.keys())}")
    ds_verify.close()
    raise ValueError("Data dimensions mismatch")
ds_verify.close()

# Output directory (in scripts/test_output)
output_dir = scripts_path / "test_output"
output_dir.mkdir(exist_ok=True)
print(f"\n📁 Output directory: {output_dir}")

# Weight directory (model/weight)
weights_dir = model_outer_path / "weight"
print(f"\n📁 Weights directory: {weights_dir}")
if weights_dir.exists():
    weight_files = list(weights_dir.glob("*.safetensors")) + list(weights_dir.glob("*.pth")) + list(weights_dir.glob("*.pt"))
    print(f"   Found {len(weight_files)} weight files")
    for f in weight_files[:5]:
        size = f.stat().st_size / (1024*1024)
        print(f"      - {f.name} ({size:.2f} MB)")
    if len(weight_files) > 5:
        print(f"      ... and {len(weight_files) - 5} more files")
else:
    print(f"   ⚠️  Weights directory not found")

# Config file
config_file = model_outer_path / "config" / "config_afnocast.yaml"
print(f"\n📁 Config file: {config_file}")
if not config_file.exists():
    print(f"   ❌ Config file not found!")
    config_dir = model_outer_path / "config"
    if config_dir.exists():
        print(f"\n   📁 Config directory contents:")
        for item in config_dir.iterdir():
            print(f"      - {item.name}")
    raise FileNotFoundError(f"Config file not found: {config_file}")
else:
    print(f"   ✅ Config file found!")

print("\n" + "=" * 60)
print("🏗️  Initializing orchestrator...")
print("=" * 60)

try:
    from inference.inference import InferenceOrchestrator
    
    orchestrator = InferenceOrchestrator(
        init_date="202505",
        config_path=str(config_file),
        trained_model_dir=str(weights_dir),
        seas_forecast_dir=str(scripts_path / "mock_data"),  # ✅ FIX: Use scripts/mock_data
        output_dir=str(output_dir),
        device=device,
        batch_size=2,
    )
    print("✅ Orchestrator initialized successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n📁 Checking Python path:")
    for i, path in enumerate(sys.path[:5], 1):
        print(f"   {i}. {path}")
    raise
except Exception as e:
    print(f"❌ Orchestrator initialization failed: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "=" * 60)
print("🏃 Running inference...")
print("=" * 60)

try:
    ds_result = orchestrator.run()
    print("✅ Inference complete!")
except Exception as e:
    print(f"❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "=" * 60)
print("📊 Results")
print("=" * 60)
print(f"Dimensions: {dict(ds_result.dims)}")
print(f"Variables: {list(ds_result.data_vars)}")

if "tp" in ds_result.data_vars:
    print(f"\n📈 tp statistics:")
    print(f"   Mean: {ds_result.tp.mean().values:.6f}")
    print(f"   Std: {ds_result.tp.std().values:.6f}")
    print(f"   Range: [{ds_result.tp.min().values:.6f}, {ds_result.tp.max().values:.6f}]")

print("\n" + "=" * 60)
print("✅ Inference pipeline complete!")
print("=" * 60)


# In[45]:


# Cell: Generate properly scaled observations
import numpy as np
import xarray as xr
import os
from pathlib import Path

def create_correctly_scaled_observations(init_date="202505", output_dir=None):
    """
    Generate correctly scaled observations
    Match the scale range of forecast data
    """
    # Get current directory
    notebook_dir = Path(os.getcwd())
    print(f"📁 Current directory: {notebook_dir}")
    
    # Determine paths correctly
    if notebook_dir.name == "seasonal_afnocast":
        project_root = notebook_dir
        model_path = project_root / "model"
        scripts_path = model_path / "scripts"
        print(f"📍 Detected: Running in project root (seasonal_afnocast)")
    elif notebook_dir.name == "scripts":
        project_root = notebook_dir.parent.parent
        model_path = notebook_dir.parent
        scripts_path = notebook_dir
        print(f"📍 Detected: Running in scripts directory")
    else:
        project_root = notebook_dir
        model_path = project_root / "model"
        scripts_path = model_path / "scripts"
        print(f"📍 Detected: Running in {notebook_dir.name}")
    
    if output_dir is None:
        output_dir = scripts_path / "mock_obs"
    
    print(f"🔄 Generating correctly scaled observations ({init_date})...")
    print(f"   Project root: {project_root}")
    print(f"   Model path: {model_path}")
    print(f"   Scripts path: {scripts_path}")
    print(f"   Output directory: {output_dir}")
    
    # Grid definition (higher resolution for observations)
    lat = np.linspace(7.1, 17.6, 106)
    lon = np.linspace(32.0, 39.7, 78)
    start_date = np.datetime64(f"{init_date[:4]}-{init_date[4:6]}-15")
    times = start_date + np.arange(-5, 6)
    
    np.random.seed(999)
    n_times, n_lat, n_lon = len(times), len(lat), len(lon)
    
    # Spatial pattern
    lat_grid, lon_grid = np.meshgrid(
        np.linspace(0, 1, n_lat),
        np.linspace(0, 1, n_lon),
        indexing='ij'
    )
    spatial = 0.5 + 0.5 * np.exp(-((lat_grid-0.5)**2 + (lon_grid-0.6)**2) * 4)
    
    # Generate observation data (scale matches forecast)
    data = np.zeros((n_times, n_lat, n_lon))
    for t in range(n_times):
        time_factor = 1 + 0.3 * np.sin(t / 11 * np.pi + 0.15)
        noise = 0.05 * np.random.randn(n_lat, n_lon)
        data[t] = np.maximum(spatial * time_factor * 2.0 + noise * 0.3, 0)
    
    # Add some extreme events
    for t in [0, 5, 8]:
        i_center = np.random.randint(30, 70)
        j_center = np.random.randint(30, 50)
        for i in range(max(0, i_center-8), min(n_lat, i_center+8)):
            for j in range(max(0, j_center-8), min(n_lon, j_center+8)):
                dist = np.sqrt((i - i_center)**2 + (j - j_center)**2)
                data[t, i, j] += 5.0 * np.exp(-dist**2 / 30)
    
    data = np.maximum(data, 0)
    
    # Save to NetCDF
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ds = xr.Dataset({
        "tp": xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lat, "lon": lon},
            attrs={
                "units": "normalized",
                "long_name": "Synthetic observations (scaled to match forecast)",
                "note": "⚠️  Synthetic data for testing only"
            }
        )
    })
    
    file_path = output_path / f"observations_{init_date}.nc"
    ds.to_netcdf(file_path)
    
    print(f"✅ Observations generated: {file_path}")
    print(f"   Dimensions: {dict(ds.sizes)}")
    print(f"   Mean: {data.mean():.6f}")
    print(f"   Min: {data.min():.6f}")
    print(f"   Max: {data.max():.6f}")
    print(f"   Non-zero: {(data > 0).sum()} / {data.size}")
    
    return file_path

# Generate correctly scaled observations
new_obs_path = create_correctly_scaled_observations("202505")

# Update observation file path
ground_truth_file = new_obs_path
print(f"\n✅ Ground truth file set to: {ground_truth_file}")
print(f"   You can now run Cell 5 (Evaluation) with this observation data")


# In[46]:


# Cell 5: Comprehensive Evaluation Metrics for Seasonal AFNOCast
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("📊 Seasonal AFNOCast - Evaluation Metrics")
print("   Bias Correction + Downscaling + Ensemble Forecast")
print("=" * 70)

# ==================== Configuration ====================
# Get current directory
notebook_dir = Path(os.getcwd())
print(f"📁 Current directory: {notebook_dir}")

# Determine paths correctly
# If we're in the project root (seasonal_afnocast)
if notebook_dir.name == "seasonal_afnocast":
    project_root = notebook_dir  # /root/private_data/seasonal_afnocast
    model_path = project_root / "model"
    scripts_path = model_path / "scripts"
    print(f"📍 Detected: Running in project root (seasonal_afnocast)")
elif notebook_dir.name == "scripts":
    project_root = notebook_dir.parent.parent  # seasonal_afnocast
    model_path = notebook_dir.parent
    scripts_path = notebook_dir
    print(f"📍 Detected: Running in scripts directory")
else:
    project_root = notebook_dir
    model_path = project_root / "model"
    scripts_path = model_path / "scripts"
    print(f"📍 Detected: Running in {notebook_dir.name}")

print(f"📁 Project root: {project_root}")
print(f"📁 Model path: {model_path}")
print(f"📁 Scripts path: {scripts_path}")

# Output directory (in scripts/test_output)
output_dir = scripts_path / "test_output"
actual_file = output_dir / "SEAS5_AFNO_v1.0_tp_202505.nc"
print(f"📁 Looking for output file: {actual_file}")

# Ground truth file (observations in scripts/mock_obs)
ground_truth_file = scripts_path / "mock_obs" / "observations_202505.nc"
# If ground truth doesn't exist, set to None
if not ground_truth_file.exists():
    print(f"⚠️  Ground truth file not found: {ground_truth_file}")
    print(f"   Running evaluation without observations (only forecast statistics)")
    ground_truth_file = None

# ==================== Load Data ====================
def load_output(filepath):
    """Load AFNOCast output"""
    ds = xr.open_dataset(filepath)
    if "tp" in ds.data_vars:
        return ds
    else:
        raise ValueError(f"No 'tp' variable found in {filepath}")

def load_ground_truth(filepath):
    """Load ground truth (ERA5-Land or observations)"""
    if filepath and filepath.exists():
        ds = xr.open_dataset(filepath)
        if "tp" in ds.data_vars:
            return ds
        else:
            print(f"⚠️  No 'tp' variable in ground truth")
            return None
    return None

# Load data
if not actual_file.exists():
    print(f"❌ Output file not found: {actual_file}")
    print(f"   Please run inference first (Cell 4)")
    print(f"   Expected location: {actual_file}")
    # Check if output directory exists
    if output_dir.exists():
        print(f"\n   📁 Output directory exists, but file not found:")
        for item in output_dir.iterdir():
            print(f"      - {item.name}")
    else:
        print(f"\n   📁 Output directory does not exist: {output_dir}")
    raise FileNotFoundError(f"Output file not found: {actual_file}")

ds_pred = load_output(actual_file)
tp_pred = ds_pred.tp  # dims: (time, ens, lat, lon)

# Get dimensions properly
print(f"\n✅ Forecast loaded:")
print(f"   Dimensions: {tp_pred.dims}")
print(f"   Shape: {tp_pred.shape}")
print(f"   Variables: {list(ds_pred.data_vars)}")

# Load ground truth if available
ds_obs = load_ground_truth(ground_truth_file)
tp_obs = ds_obs.tp if ds_obs is not None else None

if tp_obs is not None:
    print(f"✅ Observations loaded:")
    print(f"   Dimensions: {tp_obs.dims}")
    print(f"   Shape: {tp_obs.shape}")

print("\n" + "=" * 70)

# ==================== 1. Basic Statistics ====================
def compute_basic_stats(tp):
    """Compute basic statistics"""
    stats_dict = {
        "mean": float(tp.mean().values),
        "std": float(tp.std().values),
        "min": float(tp.min().values),
        "max": float(tp.max().values),
        "non_zero": int((tp > 0).sum().values),
        "non_zero_pct": float((tp > 0).sum().values / tp.size * 100),
    }
    return stats_dict

print("\n" + "=" * 70)
print("📈 1. Basic Statistics")
print("=" * 70)

pred_stats = compute_basic_stats(tp_pred)
print(f"\n🔵 Forecast (AFNOCast):")
for k, v in pred_stats.items():
    print(f"   {k}: {v}")

if tp_obs is not None:
    obs_stats = compute_basic_stats(tp_obs)
    print(f"\n🟢 Observations:")
    for k, v in obs_stats.items():
        print(f"   {k}: {v}")

# ==================== 2. Deterministic Metrics ====================
def compute_deterministic_metrics(pred, obs):
    """
    Compute deterministic metrics (requires ground truth)
    pred: forecast array
    obs: observation array
    """
    # Flatten arrays
    pred_flat = pred.values.flatten()
    obs_flat = obs.values.flatten()
    
    # Remove NaN
    valid = ~(np.isnan(pred_flat) | np.isnan(obs_flat))
    pred_flat = pred_flat[valid]
    obs_flat = obs_flat[valid]
    
    if len(pred_flat) == 0:
        return None
    
    # Metrics
    bias = float(np.mean(pred_flat - obs_flat))
    mae = float(np.mean(np.abs(pred_flat - obs_flat)))
    rmse = float(np.sqrt(np.mean((pred_flat - obs_flat)**2)))
    nrmse = rmse / (np.std(obs_flat) + 1e-8)  # Normalized RMSE
    
    # Correlation
    corr, p_value = pearsonr(pred_flat, obs_flat)
    
    # Spearman correlation (non-linear)
    spearman_corr, spearman_p = spearmanr(pred_flat, obs_flat)
    
    # Bias ratio (if obs mean > 0)
    if np.mean(obs_flat) > 1e-10:
        bias_ratio = np.mean(pred_flat) / np.mean(obs_flat)
    else:
        bias_ratio = np.nan
    
    return {
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "correlation": corr,
        "spearman_corr": spearman_corr,
        "bias_ratio": bias_ratio,
    }

if tp_obs is not None:
    print("\n" + "=" * 70)
    print("📈 2. Deterministic Metrics (Forecast vs Observation)")
    print("=" * 70)
    
    # Aggregate ensembles to ensemble mean
    pred_ens_mean = tp_pred.mean(dim="ens")
    
    print(f"\n   Forecast shape: {pred_ens_mean.shape}")
    print(f"   Observation shape: {tp_obs.shape}")
    
    # Ensure same dimensions
    if pred_ens_mean.dims != tp_obs.dims:
        print("   ⚠️  Dimension mismatch, attempting to align...")
        try:
            pred_ens_mean = pred_ens_mean.transpose(*tp_obs.dims)
        except:
            print("   ⚠️  Cannot align dimensions, skipping deterministic metrics")
            det_metrics = None
    
    try:
        det_metrics = compute_deterministic_metrics(pred_ens_mean, tp_obs)
        if det_metrics:
            print(f"\n   Bias:           {det_metrics['bias']:.6f}")
            print(f"   MAE:            {det_metrics['mae']:.6f}")
            print(f"   RMSE:           {det_metrics['rmse']:.6f}")
            print(f"   NRMSE:          {det_metrics['nrmse']:.4f}")
            print(f"   Correlation:    {det_metrics['correlation']:.4f}")
            print(f"   Spearman:       {det_metrics['spearman_corr']:.4f}")
            if not np.isnan(det_metrics['bias_ratio']):
                print(f"   Bias Ratio:     {det_metrics['bias_ratio']:.4f}")
        else:
            print("   ⚠️  Could not compute metrics")
    except Exception as e:
        print(f"   ⚠️  Error computing metrics: {e}")
        det_metrics = None
else:
    det_metrics = None
    print("\n" + "=" * 70)
    print("📈 2. Deterministic Metrics")
    print("=" * 70)
    print("   ⚠️  No ground truth available. Skipping deterministic metrics.")
    print("   To enable: Run Cell to generate observations first")

# ==================== 3. Spatial Pattern Metrics ====================
def compute_spatial_metrics(pred, obs):
    """Compute spatial pattern metrics"""
    # Time-averaged patterns
    pred_pattern = pred.mean(dim="time")
    obs_pattern = obs.mean(dim="time")
    
    # Flatten
    p_flat = pred_pattern.values.flatten()
    o_flat = obs_pattern.values.flatten()
    
    # Remove NaN
    valid = ~(np.isnan(p_flat) | np.isnan(o_flat))
    p_flat = p_flat[valid]
    o_flat = o_flat[valid]
    
    if len(p_flat) < 3:
        return None
    
    # Pattern correlation
    pat_corr, _ = pearsonr(p_flat, o_flat)
    
    # Spatial bias
    spatial_bias = float(np.mean(p_flat - o_flat))
    
    # Spatial variance ratio
    var_ratio = np.var(p_flat) / (np.var(o_flat) + 1e-8)
    
    return {
        "pattern_correlation": pat_corr,
        "spatial_bias": spatial_bias,
        "variance_ratio": var_ratio,
    }

if tp_obs is not None:
    print("\n" + "=" * 70)
    print("📈 3. Spatial Pattern Metrics")
    print("=" * 70)
    
    try:
        pred_ens_mean = tp_pred.mean(dim="ens")
        spatial_metrics = compute_spatial_metrics(pred_ens_mean, tp_obs)
        if spatial_metrics:
            print(f"\n   Pattern Correlation: {spatial_metrics['pattern_correlation']:.4f}")
            print(f"   Spatial Bias:        {spatial_metrics['spatial_bias']:.6f}")
            print(f"   Variance Ratio:      {spatial_metrics['variance_ratio']:.4f}")
            
            # Interpretation
            print(f"\n   💡 Interpretation:")
            print(f"      Pattern Corr > 0.7 = Good spatial pattern match")
            if spatial_metrics['variance_ratio'] > 0.8 and spatial_metrics['variance_ratio'] < 1.2:
                print(f"      Variance Ratio ~1.0 = Good amplitude match")
            elif spatial_metrics['variance_ratio'] < 0.8:
                print(f"      Variance Ratio < 1.0 = Under-dispersed (smoothed)")
            else:
                print(f"      Variance Ratio > 1.2 = Over-dispersed (too variable)")
        else:
            print("   ⚠️  Not enough data for spatial metrics")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
else:
    print("\n" + "=" * 70)
    print("📈 3. Spatial Pattern Metrics")
    print("=" * 70)
    print("   ⚠️  No ground truth available. Skipping spatial metrics.")

# ==================== 4. Precipitation Event Metrics ====================
def compute_event_metrics(pred, obs, threshold=1.0):
    """
    Compute event detection metrics (mm/day)
    threshold: precipitation threshold (default 1 mm/day)
    """
    # Apply threshold
    pred_binary = (pred >= threshold).astype(float)
    obs_binary = (obs >= threshold).astype(float)
    
    # Flatten
    p_flat = pred_binary.values.flatten()
    o_flat = obs_binary.values.flatten()
    
    # Remove NaN
    valid = ~(np.isnan(p_flat) | np.isnan(o_flat))
    p_flat = p_flat[valid]
    o_flat = o_flat[valid]
    
    # Confusion matrix
    tp_event = np.sum((p_flat == 1) & (o_flat == 1))
    fp_event = np.sum((p_flat == 1) & (o_flat == 0))
    fn_event = np.sum((p_flat == 0) & (o_flat == 1))
    tn_event = np.sum((p_flat == 0) & (o_flat == 0))
    
    # Metrics
    pod = tp_event / (tp_event + fn_event + 1e-8)
    far = fp_event / (tp_event + fp_event + 1e-8)
    csi = tp_event / (tp_event + fp_event + fn_event + 1e-8)
    freq_bias = (tp_event + fp_event) / (tp_event + fn_event + 1e-8)
    
    # ETS
    random_hits = (tp_event + fp_event) * (tp_event + fn_event) / (tp_event + fp_event + fn_event + tn_event + 1e-8)
    ets = (tp_event - random_hits) / (tp_event + fp_event + fn_event - random_hits + 1e-8)
    
    return {
        "threshold": threshold,
        "pod": pod,
        "far": far,
        "csi": csi,
        "freq_bias": freq_bias,
        "ets": ets,
        "hits": int(tp_event),
        "false_alarms": int(fp_event),
        "misses": int(fn_event),
    }

if tp_obs is not None:
    print("\n" + "=" * 70)
    print("📈 4. Precipitation Event Metrics")
    print("=" * 70)
    print("   Evaluating ability to detect precipitation events")
    print("   " + "-" * 60)
    print("   " + "Confusion Matrix: Hits = Correctly predicted rain")
    print("   " + "                  False Alarms = False rain prediction")
    print("   " + "                  Misses = Missed rain events")
    
    thresholds = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
    pred_ens_mean = tp_pred.mean(dim="ens")
    
    print(f"\n   {'Thresh (mm/day)':<18} {'POD':<8} {'FAR':<8} {'CSI':<8} {'ETS':<8} {'Bias':<8} {'Hits':<8}")
    print("   " + "-" * 75)
    
    event_results = []
    for thresh in thresholds:
        try:
            event_metrics = compute_event_metrics(pred_ens_mean, tp_obs, thresh)
            if event_metrics:
                print(f"   {thresh:>5.1f} mm/day     "
                      f"{event_metrics['pod']:>6.3f}  "
                      f"{event_metrics['far']:>6.3f}  "
                      f"{event_metrics['csi']:>6.3f}  "
                      f"{event_metrics['ets']:>6.3f}  "
                      f"{event_metrics['freq_bias']:>6.3f}  "
                      f"{event_metrics['hits']:>6}")
                event_results.append(event_metrics)
        except Exception as e:
            print(f"   {thresh:>5.1f} mm/day     Error: {e}")
    
    # Interpretation guide
    print("\n   💡 Interpretation:")
    print("      POD > 0.7 = Good at detecting rain events")
    print("      FAR < 0.3 = Low false alarm rate")
    print("      CSI > 0.5 = Good overall performance")
    print("      ETS > 0.3 = Good skill (accounts for random hits)")
    print("      Bias near 1.0 = Correct frequency bias")
else:
    print("\n" + "=" * 70)
    print("📈 4. Precipitation Event Metrics")
    print("=" * 70)
    print("   ⚠️  No ground truth available. Skipping event metrics.")
    print("   To enable: Run Cell to generate observations first")

# ==================== 5. Summary and Save ====================
print("\n" + "=" * 70)
print("📊 5. Summary")
print("=" * 70)

# Compile all metrics
all_metrics = {
    "forecast_stats": pred_stats,
    "file_info": {
        "name": actual_file.name,
        "size_mb": actual_file.stat().st_size / (1024 * 1024),
        "dimensions": list(tp_pred.dims),
        "shape": list(tp_pred.shape),
    }
}

if det_metrics:
    all_metrics["deterministic_metrics"] = det_metrics

if tp_obs is not None:
    all_metrics["observation_stats"] = obs_stats
    
    # Try to add event metrics summary
    try:
        all_metrics["event_metrics"] = event_results
    except:
        pass

# Save metrics to file (in scripts/test_output/metrics)
metrics_dir = scripts_path / "test_output" / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
metrics_file = metrics_dir / "evaluation_metrics.json"
with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)
print(f"\n💾 Metrics saved to: {metrics_file}")

print("\n" + "=" * 70)
print("✅ Evaluation complete!")
print("=" * 70)

# Close datasets
ds_pred.close()
if ds_obs:
    ds_obs.close()


# In[ ]:




