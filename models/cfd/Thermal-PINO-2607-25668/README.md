# Thermal-PINO-2607-25668

Tier 1 runnable reproduction artifact for arXiv:2607.25668, based on the paper's 1-D transient wall heat equation, Crank-Nicolson data generation, and parameter-to-temperature operator formulation.

This package uses a NumPy Fourier-feature fallback because the packaging environment does not provide PyTorch. It is a reproducibility scaffold and does not claim the paper's reported PINO metrics.

## Files

- `model/`: Fourier operator implementation
- `weight/`: Tier 1 fitted checkpoint
- `scripts/`: training, evaluation, and smoke-test entry points
- `conf/`: configuration and audit metadata

## Usage

```bash
python scripts/smoke_test.py
python scripts/train.py
python scripts/evaluate.py
```

Paper: https://arxiv.org/abs/2607.25668

