# Operator Boosting PDE 2606.17460

Self-contained PyTorch reproduction of the normalized residual operator boosting protocol described in arXiv:2606.17460. The implementation includes parameter-free `G_0`, independently fitted residual stages, validation selection of shrinkage coefficients including `eta=0`, and a separate same-family full operator baseline in the checkpoint.

## Important metric status

The bundled configuration uses a labeled synthetic 1D PDE-like field because the paper's benchmark data and several benchmark protocol details are unavailable. Any metrics produced with this configuration are **synthetic fallback benchmark results, not paper benchmark results and not claims of reproducing the paper's tables**. Replace `conf/config.json` with a real dataset configuration before making scientific comparisons.

## Contents

- `model/`: model, operator, data, loss, and configuration implementation.
- `weight/checkpoint.pt`: Tier 1 trained checkpoint.
- `scripts/train.py`: train and write a checkpoint.
- `scripts/infer.py`: predict from a `torch.save` input tensor.
- `scripts/evaluate.py`: evaluate on the configured test split.
- `conf/config.json`: synthetic fallback configuration and data schema.

## Usage

Install Python and PyTorch, then run from this model directory:

```bash
python scripts/train.py --output /tmp/operator_boosting.pt
python scripts/evaluate.py
python scripts/infer.py --input input.pt --output prediction.pt
```

External data must be a `torch.save` file containing `train`, `validation`, and `test` pairs `(a, u)` with tensors shaped `[B, C, *spatial_shape]`, as configured. Training outputs a checkpoint compatible with the inference and evaluation scripts.

## Citation

```bibtex
@misc{operatorboosting2026,
  title         = {Operator Boosting},
  year          = {2026},
  eprint        = {2606.17460},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2606.17460}
}
```

## License

Apache License 2.0. This package is a OneScience reproduction artifact and does not upload or publish any external data.

