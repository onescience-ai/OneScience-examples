<p align="center"><strong><span style="font-size: 30px;">Shodh-MoE-2605-15179</span></strong></p>

# Model Introduction

Tier 1 quick reproduction artifact for *Eradicating Negative Transfer in Multi-Physics Foundation Models via Sparse Mixture-of-Experts Routing*. It packages a configurable PyTorch 3-D tokenizer, constrained decoder, shared block, Top-1 routed experts, checkpoint, and validation telemetry derived from the supplied paper reproduction materials.

# Model Description

- `model/model.py`: 3-D encoder, curl-based divergence-free velocity decoder, shared path, Top-1 router, routed experts, and load-balance telemetry.
- `model/config.py`: explicit configurable dimensions, loss weights, optimizer, seed, and distributed settings.
- `model/data.py`: deterministic synthetic mixed-PDE stand-in with two telemetry domains.
- `model/train.py` and `model/metrics.py`: training loss, validation, reconstruction, routing, and FP64 divergence metrics.

# Intended Use

| Scenario | Entry point |
| --- | --- |
| Tier 1 synthetic training | `python scripts/tier1_train.py` |
| CPU smoke test with forward/backward/train/validation checks | `python scripts/smoke.py` |
| Checkpoint-backed evaluation | `python scripts/eval.py` |

# Usage

## OneCode

This package is compatible with the OneCode workflow. Use the included entry points and configuration as the model package interface.

## Manual Installation

Install a compatible Python 3 environment with PyTorch. GPU execution requires a matching CUDA-enabled PyTorch build; CPU execution is supported for the reduced smoke path. The original paper reports a 32-H100 distributed setup, but this package does not reproduce that resource configuration.

### Download model package

```bash
modelscope download --model OneScience/Shodh-MoE-2605-15179
```

### Train

```bash
python scripts/tier1_train.py
```

### Train checkpoint

The packaged checkpoint is `weight/shodh_moe_tier1.pt`, recorded at training step 200. Metrics are in `conf/metrics.json`.

### Evaluate

```bash
python scripts/eval.py
```

### Smoke test

```bash
python scripts/smoke.py
```

# Exact-Paper Limitations

This is a successful Tier 1 synthetic quick reproduction, not an exact reproduction of the paper. The paper's approximately 61,000-sample Zarr corpus, schema, normalization, full `128^3 -> 16^3` production configuration, exact widths and layer order, optimizer, loss formula, second-order alignment objective, checkpoint, split, Triton kernels, 20,000-step distributed H100 run, and reported experimental metrics were unavailable or unspecified. This package uses reduced `8^3 -> 4^3` spatial dimensions, synthetic data, explicit local assumptions, 200 training steps, and a single-process CPU-compatible path. No temporal rollout or exact paper metric is claimed.

# Validation

The supplied Tier 1 validation report records finite physical and relative MSE, active routing for both experts, load-balance telemetry, near-zero FP64 decoded velocity divergence, and successful checkpoint reload. See `conf/validation_report.md` and `conf/metrics.json`.

# Citation and License

```text
Ellwil Sharma and Arastu Sharma (Shodh AI), "Eradicating Negative Transfer in Multi-Physics Foundation Models via Sparse Mixture-of-Experts Routing," arXiv:2605.15179, 2026.
https://arxiv.org/abs/2605.15179
```

The package is released under Apache License 2.0 for this reproduction artifact. Paper rights and data licenses remain with their respective authors and providers.

