<p align="center"><strong><span style="font-size: 30px;">MiNO-2608-15187</span></strong></p>

# Model Introduction

This package is a Tier 1 reproduction of **MiNO: Cotangent-bundle propagator learning for PDEs** (arXiv:2608.15187). It learns a phase-amplitude propagator for one-dimensional smooth advection and reconstructs the solution with a Fourier integral.

# Model Description

- `model/model.py`: phase and amplitude networks, eikonal/transport residuals, and Fourier reconstruction.
- Phase parameterization: `Phi=x*xi+t*h_theta`.
- Amplitude parameterization: `A=1+t*gamma_theta`.
- PDE calibration: `u_t + u_x = 0` with analytic Gaussian initial data.

# Usage

## Training

```bash
python scripts/train.py --steps 2000 --collocation 4096 --output-dir artifacts
```

## Evaluation and Smoke Test

```bash
python scripts/smoke_test.py
```

The included checkpoint was trained with one seed and 2000 optimizer steps. Its final-time test relative L2 is `0.03337915259720496`.

# Hardware

The included Tier 1 run works on CPU. GPU execution is supported by passing `--device cuda` when a compatible PyTorch installation is available.

# Data

No external dataset is required. Samples are generated analytically from the smooth-advection PDE, with `x in [-3,3]`, `t in [0,1.5]`, and `xi in [-30,30]`.

# Limitations

This is a reduced Tier 1 reproduction, not the paper's 50,000-step five-seed aggregate. The paper does not specify the hidden activation, exact modified-MLP skip formula, feature packing, dtype, transport weight, or discrete evaluation grid; this package records reversible implementation assumptions in the source README and reproduction specification.

# Citation

```text
Gnankan Landry Regis N'guessan and Bum Jun Kim. MiNO: Cotangent-bundle propagator learning for PDEs. arXiv:2608.15187, 2026.
https://arxiv.org/abs/2608.15187
```

