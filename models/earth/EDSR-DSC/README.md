# EDSR-DSC

EDSR-DSC is a four-times super-resolution model for downscaling two-channel
wind-velocity fields. It takes coarse `u` and `v` components and produces
higher-resolution fields with four times the input height and width.

- Official model: https://huggingface.co/lschmidt/edsr-dsc
- Task: wind-field spatial downscaling
- Input channels: 2 (`u`, `v`)
- Output channels: 2 (`u`, `v`)
- Scale factor: 4

## Files

- `train.py`: self-contained functional validation and inference script.
- `config.json`: official model configuration.
- `download_resources.sh`: downloads and verifies the official checkpoint and
  sample NetCDF data.
- `requirements.txt`: non-framework Python dependencies.

Model weights, NetCDF data, generated validation results, and test reports are
not stored in Git.

## Shared resources

OneScience Notebook paths:

```text
/root/group_data/SDU-Test/EDSR-DSC/pytorch_model_4x.pt
/root/group_data/SDU-Test/EDSR-DSC/test_data/test_wind_velocities.nc
```

SCNet paths:

```text
/public/share/sugonhpcapp01/SDU-Test/EDSR-DSC/pytorch_model_4x.pt
/public/share/sugonhpcapp01/SDU-Test/EDSR-DSC/test_data/test_wind_velocities.nc
```

## Download resources

For environments without access to the shared directory:

```bash
bash download_resources.sh
```

An alternative output directory can be supplied as the first argument:

```bash
bash download_resources.sh /path/to/edsr-dsc
```

The script verifies both files using their expected SHA256 hashes.

## Functional validation

In the OneScience Notebook container:

```bash
python train.py --device cuda \
  --weights /root/group_data/SDU-Test/EDSR-DSC/pytorch_model_4x.pt \
  --data /root/group_data/SDU-Test/EDSR-DSC/test_data/test_wind_velocities.nc
```

If no NetCDF input is available, deterministic synthetic input can be forced:

```bash
python train.py --device cuda \
  --weights /path/to/pytorch_model_4x.pt \
  --data none
```

The completed OneScience test passed checkpoint loading and CUDA inference.
Because the available local NetCDF file does not contain a paired
high-resolution target, the reported result is a functional validation rather
than a scientific-accuracy evaluation.
