# SEAM-Ω Tier 1

Finite explanation-sheaf audit reproduction for *SEAM: Global consistency beyond local accuracy in scientific machine learning* (arXiv:2608.05702).

## Contents

- `model/seam.py`: SEAM-Ω audit API.
- `scripts/train_tier1.py`: deterministic synthetic end-to-end entry point.
- `conf/config.json`: model configuration.
- `weight/model_weights.npz`: Tier 1 serialized NumPy weights/configuration.

The package computes overlap obstruction `omega = D s`, channel-resolved norms, exact budget feasibility, residual-aware repairs, and identifiability rank. It does not claim full reproduction of all nineteen paper experiments.

## Usage

```bash
python scripts/train_tier1.py
```

## Citation

N'guessan, G. L. R. and Kim, B. J. SEAM: Global consistency beyond local accuracy in scientific machine learning. arXiv:2608.05702, 2026.

