# DRIFT Tier 1

Minimal PyTorch reproduction of the paper's Distributed Truncated Spectral Transform (DTST) and DRIFT spectral block, based only on arXiv:2607.14394v1.

The included run uses a deterministic synthetic periodic PDE fallback. It is not PDEBench and its metrics must not be interpreted as paper benchmark results. The CPU smoke test measured reduced-DFT equivalence at relative error `1.88e-15`. No multi-GPU or speedup evidence is included.

Entrypoints are in `scripts/`; model code is in `model/`; configuration is in `conf/`; the trained checkpoint is in `weight/`.

