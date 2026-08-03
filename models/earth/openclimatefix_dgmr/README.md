# DGMR inference reproduction

This directory reproduces single-sample inference for the exact Hugging Face
model repository:

```text
openclimatefix/dgmr
```

The implementation package is the Open Climate Fix `dgmr` Python package.
The upstream project documents loading the complete model with:

```python
from dgmr import DGMR
model = DGMR.from_pretrained("openclimatefix/dgmr")
```

## Files

```text
dgmr/
├── .gitignore
├── README.md
├── environment.txt
├── reproduce_dgmr.py
└── requirements.txt
```

The large Hugging Face model weight is downloaded during execution and is not
committed to Gitee.

## Environment

See `environment.txt`.

Retain the OneScience / FlagOS platform-provided PyTorch, TorchVision and
FlagGems builds. Do not replace them with public PyPI builds.

## Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps dgmr==1.4.4
```

## Run

Native PyTorch:

```bash
python reproduce_dgmr.py
```

Optional FlagGems compatibility test:

```bash
python reproduce_dgmr.py --flag-gems
```

Generate multiple stochastic ensemble members:

```bash
python reproduce_dgmr.py --members 3
```

## Required identity check

A valid run for this submission must print:

```text
Hugging Face 模型： openclimatefix/dgmr
```

A run that prints `openclimatefix-models/dgmr` is a different repository and
must not be used as the acceptance result for this submission.

The local Hugging Face cache should include:

```text
hf_cache/hub/models--openclimatefix--dgmr
```

## Input

The script generates a synthetic radar history:

```text
shape = (1, 4, 1, 256, 256)
dtype = float32
```

## Output

For the default single ensemble member, the expected prediction is:

```text
shape = (1, 18, 1, 256, 256)
dtype = float32
```

Generated files are written to `dgmr_outputs/`, including NPY arrays, PNG
images, a GIF animation and `reproduction_report.json`.

A successful engineering test requires the expected shape and no NaN or Inf.

## Scope

The synthetic input validates model download, deserialization and forward
inference. It does not reproduce paper-level forecasting accuracy, and the raw
network output must not be interpreted directly as a calibrated rainfall rate.
