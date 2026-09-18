# LIGO-PINN

Tier 1 reproduction of *LIGO-PINN: Learned Initialization via Gated Optimization to Alleviate Convergence Failures in Physics Informed Neural Networks* (arXiv:2607.14233). This package implements the paper's invariance encoding and gated layer-wise optimization on the fully specified 1D convection benchmark.

## Contents

- `model/`: PINN architecture and PDE helpers.
- `weight/model.pt`: trained Tier 1 weights.
- `scripts/train.py`: training entry point.
- `scripts/evaluate.py`: evaluation entry point reference.
- `scripts/smoke_test.py`: six-check smoke test.
- `conf/`: reproducibility configuration and metrics.

## Usage

```bash
python scripts/train.py --output-dir artifacts/ligo_pinn
python scripts/smoke_test.py
```

The Tier 1 run uses easy beta tasks `{5,10,15,20,25}` for learned initialization and evaluates the hard task `beta=40` with reduced collocation points and iterations. The resulting MAE is traceable in `conf/evaluation_metrics.json`.

## Citation

Anurag, N., Adhikari, S., Kapoor, T., and Muralidhar, N. LIGO-PINN, arXiv:2607.14233, 2026. https://arxiv.org/abs/2607.14233

