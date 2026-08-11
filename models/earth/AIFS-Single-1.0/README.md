# ECMWF AIFS Single v1.0

AIFS Single v1.0 is ECMWF's data-driven global weather forecasting model. This
example validates the official checkpoint with the official Anemoi inference
runner in the OneScience Notebook environment.

- Official model: https://huggingface.co/ecmwf/aifs-single-1.0
- Task: global medium-range weather forecasting
- Grid: N320 reduced Gaussian grid
- Forecast step used by this test: 6 hours
- Model parameters: 253,035,398

## Files

- `train.py`: checkpoint verification, environment preflight and functional
  forecast validation.
- `config_pretraining.yaml`: official pretraining configuration.
- `config_finetuning.yaml`: official fine-tuning configuration.
- `model_manifest.json`: pinned repository revision, size and SHA256 metadata.
- `download_checkpoint.sh`: downloads and verifies the official checkpoint.
- `requirements.txt`: additional non-framework dependencies.

The checkpoint, generated NPZ arrays, validation results, environment snapshots
and test reports are not stored in Git.

## Shared checkpoint

OneScience Notebook:

```text
/root/group_data/SDU-Test/aifs-single-1.0/aifs-single-mse-1.0.ckpt
```

SCNet:

```text
/public/share/sugonhpcapp01/SDU-Test/aifs-single-1.0/aifs-single-mse-1.0.ckpt
```

## Download the checkpoint

For environments without access to the shared directory:

```bash
bash download_checkpoint.sh
```

An alternative output path can be supplied as the first argument:

```bash
bash download_checkpoint.sh /path/to/aifs-single-mse-1.0.ckpt
```

The script verifies the exact checkpoint size and SHA256 before accepting it.

## Environment

The OneScience container supplies the deep-learning stack, including PyTorch,
Flash Attention and torch-geometric. Do not reinstall those packages.

The test used:

- Python 3.10.12
- PyTorch 2.4.1
- `anemoi-inference==0.4.9`
- `anemoi-models==0.3.1`
- `anemoi-utils[provenance,text]==0.4.9`
- container-provided `torch-geometric==2.7.0`
- container-provided `flash-attn==2.6.1`

Install the two Anemoi packages without dependency resolution so that pip does
not replace the container's deep-learning stack:

```bash
python -m pip install --no-deps \
  anemoi-inference==0.4.9 \
  anemoi-models==0.3.1
```

Then install the additional non-framework dependencies recorded in
`requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Use the preflight check before inference:

```bash
python train.py --device cuda \
  --checkpoint /root/group_data/SDU-Test/aifs-single-1.0/aifs-single-mse-1.0.ckpt \
  --preflight-only
```

## Functional validation

```bash
python train.py --device cuda --chunks 16 \
  --checkpoint /root/group_data/SDU-Test/aifs-single-1.0/aifs-single-mse-1.0.ckpt
```

The completed public-path validation produced:

- Status: `PASS`
- Validation level: `functional_only`
- Data source: `synthetic_n320_weather_like`
- Checkpoint SHA256 verified: `True`
- Official Anemoi runner loaded: `True`
- Input fields: `94`
- Input field shape: `(2, 542080)`
- Forecast steps: `1`
- Output fields: `102`
- Output field shape: `(542080,)`
- Finite-output ratio: `1.000000`
- Median latency: `14.031 s`
- Peak GPU memory: `22518.73 MB`

The container's optional `torch-scatter` and `torch-cluster` extensions emitted
CUDA-runtime warnings and were disabled. The checkpoint still loaded and the
forecast completed using the compatibility adapter in `train.py`.

The test uses deterministic synthetic weather-like input. It validates
checkpoint integrity, official-runner loading and functional inference only; it
does not establish scientific forecast accuracy.
