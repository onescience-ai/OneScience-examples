# GNO-Parametric-BVP-2607-21932

Tier 1 implementation of **Generalized Neural Operator for Parametric and Boundary-Value Problems**, arXiv:2607.21932. The package targets a small synthetic 2D Heat problem and includes explicit PDE parameter conditioning, a hard-gated kernel mixture, boundary transfer attention, and group-DRO training.

## Contents

- `model/`: GNO model, synthetic Heat data adapter, and nMSE metric.
- `weight/best_model.pt`: Tier 1 checkpoint.
- `scripts/train.py`: configurable training entrypoint.
- `scripts/evaluate.py`: checkpoint evaluation entrypoint.
- `scripts/smoke_test.py`: six-check smoke test.
- `conf/tier1.yaml`: reproducible small-data configuration.

## Usage

```bash
PYTHONPATH=model python scripts/train.py --config conf/tier1.yaml
PYTHONPATH=model python scripts/evaluate.py --checkpoint weight/best_model.pt --config conf/tier1.yaml
PYTHONPATH=model python scripts/smoke_test.py --config conf/tier1.yaml
```

The small-data defaults are an engineering reproduction target and are not the paper's full multi-PDE results. Unspecified paper implementation details are exposed as configuration assumptions.

## Citation

Li, Ruoyan, Yizhou Sun, and Wei Wang. "Generalized Neural Operator for Parametric and Boundary-Value Problems." arXiv:2607.21932, 2026. https://arxiv.org/abs/2607.21932

