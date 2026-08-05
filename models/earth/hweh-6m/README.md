# Hweh-6M inference reproduction

This directory reproduces single-sample inference for:

```text
Harley-ml/Hweh-6M
```

The model is pinned to the reviewed Hugging Face commit:

```text
710d8df83c41ed4555cd94871f8b64382f514bb8
```

Hweh-6M is a multitask LSTM weather model that consumes 72 hourly records
with 22 features and predicts 12 hours of multivariate weather.

## Files

```text
hweh-6m/
├── .gitignore
├── README.md
├── environment.txt
├── reproduce_hweh.py
└── requirements.txt
```

The script downloads `config.json`, `configuration.py`, `modeling.py` and
`model.safetensors` automatically. Downloaded model files and generated
outputs are not committed to Gitee.

## Compatibility repair

The pinned upstream `configuration.py` references
`distill_teacher_head_dim` without defining it. The reproduction script
automatically applies the tested compatibility rule:

```text
read distill_teacher_head_dim from kwargs;
use hidden_dim as the default value.
```

It also uses and resets a project-local Transformers dynamic-module cache so
that stale custom code is not reused.

## Environment

See `environment.txt`. Retain the platform-provided PyTorch and FlagGems
builds; do not replace them with public PyPI builds.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

Native PyTorch:

```bash
python reproduce_hweh.py
```

Optional FlagGems test:

```bash
python reproduce_hweh.py --flag-gems
```

Optional recent Open-Meteo history:

```bash
python reproduce_hweh.py   --data-mode open-meteo   --latitude 47.6062   --longitude -122.3321   --location-index 0
```

The default synthetic mode requires no external weather-data file.

## Expected engineering result

A successful default run should report:

```text
Hugging Face 模型： Harley-ml/Hweh-6M
输入特征形状： (72, 22)
预测行数： 12
输出全部有限： True
```

Generated files are written to `hweh_outputs/`, including normalized input,
12-hour forecast tables, raw model outputs, plots and
`reproduction_report.json`.

A warning that `distill_proj.weight` was newly initialized may appear because
that tensor is absent from the checkpoint. This does not by itself mean that
the 12-task forward pass failed, but the synthetic test is an engineering
compatibility check rather than a validation of real-world forecast accuracy.

## Scope

The model is not intended for safety-critical weather forecasting or as a
replacement for an operational meteorological service.
