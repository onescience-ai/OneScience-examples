<p align="center"><strong><span style="font-size: 30px;">PACE-FNO-2605-18606</span></strong></p>

# Model Introduction

Tier 1 reduced-scale reproduction of Physics-Aligned Canonical Equivariant Fourier Neural Operator under Symmetry-Induced Shifts (arXiv:2605.18606). The model estimates a Lie-algebra frame, applies a spectral canonicalization, predicts with a standard FNO, and restores the terminal frame.

# Model Description

- `model/pace_fno.py`: PACE-FNO encoder, spectral translation, and FNO implementation.
- `weight/pace_fno_tier1.pt`: Tier 1 checkpoint.
- `conf/config.json`: reproducibility configuration and dataset contract.

# Usage

```bash
python scripts/smoke_test.py
python scripts/train.py --epochs 30
```

The package uses a deterministic synthetic periodic 1-D Burgers benchmark with 32/8/8 samples and a 64-point grid. The reported Tier 1 test relative L2 is 0.6963119506835938. This reduced run is not an exact reproduction of the full paper tables.

# Citation

Xu, J., Mou, C., Zhang, Y., He, F. Physics-Aligned Canonical Equivariant Fourier Neural Operator under Symmetry-Induced Shifts. arXiv:2605.18606, 2026.
