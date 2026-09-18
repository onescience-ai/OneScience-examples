# CSNO Reproduction — Cellular MHD Sheaf Neural Operator

Reproduction code for *Cellular Sheaf Neural Operators for Structure-Preserving
Surrogate Modeling of Constrained PDEs* (arXiv:2606.00937).  This implements the
primary deliverable, `CellularMHDSheafNeuralOperator` (experiment key
`sheaf_mhd`), plus the `unet3d` / `fno3d` / `mlp` / `sheaf_equilibrium`
baselines, for The Well `MHD_64` and ConStellaration.

## Layout

```
src/complexes/     oriented cell complex, cochains, Hodge, incidence (sparse)
src/sheaves/       sheaf restriction maps, message block, Hodge Laplacian
src/models/        cellular MHD SNO, sheaf_mhd alias, UNet3D, FNO3D, MLPs
src/datasets/      WellMHD64Dataset, ConStellarationDataset
src/physics/       finite differences, divergence, curl, spectra, MHD metrics
src/training/      mhd_loss, Trainer, evaluate, rollout_evaluate
src/utils/         config, checkpoint, json/jsonl, seeding
configs/           default_experiment.yaml, smoke_experiment.yaml
experiments.py     primary driver
smoke_test.py      Tier-0 full-path smoke test (6 checks)
```

## Behaviour

Magnetic flux is a **face 2-cochain** updated exactly by
`B_next = B - dt d1 E`, so `d2 B_next = d2 B` (because `d2 d1 = 0`) — the
magnetic divergence constraint is preserved *structurally*, independent of
weights/optimizer.  Fluid state is a **volume 3-cochain** updated by
`U_next = U - dt d2 F + dt S`.

## Running

Environment (DCU node):

```bash
export PATH=/public/software/apps/anaconda3/2023.09/bin:$PATH && conda activate onescience311
module load compiler/dtk/25.04.4
```

Tier-0 smoke (synthetic data, no downloads):

```bash
sbatch --wait jobs/smoke_job.sh   # or: python smoke_test.py on the compute node
```

Full training (requires The Well MHD_64 HDF5 placed under the configured
`data_root`, and ConStellaration JSONL subset for the optional track):

```bash
SMOKE_TEST=0 python experiments.py           # FINAL_RUN (10 seeds, 20/100 epochs)
FAST_DEV_RUN=1 python experiments.py         # quick dev suite
```

The driver writes per-seed artifacts under
`outputs/<prefix>_experiment_<timestamp>/runs/<dataset>/<model>/seed_<seed>/`:
`config_resolved.json`, `train_log.csv`, `epoch_timing.csv`, `best_model.pt`,
`last_model.pt`, `metrics_valid.json`, `metrics_test.json`,
`rollout_metrics.json`, `complete_run.json`/`failed_run.json`.

## DCU adaptation

- AMP uses `bfloat16`; `torch.compile` is default-off for `sheaf_mhd` /
  `cellular_mhd_sno` (matches official `model_compile_defaults`).
- Sparse incidence matmul disables autocast and computes in float32 (DCU sparse
  `addmm` has no low-precision kernels).
- Device is reported as `cuda` by PyTorch on DCU; the code falls back to CPU when
  no accelerator is present (useful for a functional smoke without GPUs).

## Known gaps / assumptions

- The Well MHD_64 and ConStellaration data are **not** downloaded by this repo;
  point each dataset's `data_root` at local files (see `configs/`).
- Target metrics are validated against the paper report (MSE≈1.234,
  RelL2≈0.388, magnetic-divergence L2≈0.235, mean rollout RelL2≈0.486);
  righter alignment relative to DCU-vs-NVIDIA reproducibility is expected within
  a reasonable range.