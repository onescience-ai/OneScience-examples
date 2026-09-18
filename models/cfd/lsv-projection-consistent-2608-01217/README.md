# LSV Projection-Consistent Neural Operator

Compact Tier 1 reproduction of *Amortizing the Calibration Triple: A Projection-Consistent Neural Operator for Local-Stochastic Volatility* (arXiv:2608.01217).

The package predicts implied volatility, Dupire local variance, the SV conditional moment, and leverage variance with the structural relation `ell2 * m = a_D`. Data are deterministic synthetic teacher states because the paper does not provide a dataset artifact. Paper-result parity is not claimed.

## Contents

- `model/`: Conditional DeepONet implementation.
- `weight/`: Tier 1 PyTorch checkpoint.
- `scripts/`: training and metric inspection entry points.
- `conf/`: package metadata and trusted-domain configuration.

## Usage

```bash
python scripts/train.py
python scripts/evaluate.py
```

Paper: https://arxiv.org/abs/2608.01217

