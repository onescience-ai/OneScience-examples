# STCO: Conditional Neural Operators

Tier 1 runnable reproduction of `STCO: Conditional Neural Operators for Time-Dependent PDEs` (arXiv:2608.20477). The package implements FAGL-style vorticity-aware slots, four-neighbor IDW alignment, and dual-site DSFiLM for prescribed target-time conditions.

This deliverable uses a small deterministic synthetic time-dependent PDE dataset for end-to-end validation. It is an implementation/interface reproduction and does not claim the paper's full 142-simulation WaterLily benchmark numbers.

## Layout

- `model/stco_model.py`: STCO model interface.
- `weight/model.pt`: Tier 1 trained weights.
- `scripts/train.py`: training entry point.
- `scripts/evaluate.py`: relative-L2 and condition sensitivity evaluation.
- `scripts/smoke_test.py`: Tier 0 six-check smoke test.
- `conf/`: configuration, metrics, and smoke-test evidence.

## Usage

From the repository root with PyTorch installed:

```bash
python scripts/smoke_test.py
python scripts/train.py
python scripts/evaluate.py
```

The paper defines response channels `[u, v, p]` and condition channels `[psi, delta_psi, g_x, g_y, u_bc_x, u_bc_y]`. Full reproduction would require the WaterLily data and complete benchmark parameter tables, which are recorded as gaps in `.paper2code_work/2608.20477/reproduction_spec.md`.

