# Local Model Source

This directory contains OneScience `models` sources that ESM imports directly.

- `esm/`: local copy of `onescience.models.esm`
- `openfold/`: local copy of `onescience.models.openfold`, required by ESMFold
- `protenix/layer_norm/`: optional fused layer norm dependency used by OpenFold when `LAYERNORM_TYPE=fast_layernorm`

Other OneScience namespaces such as `onescience.datapipes`, `onescience.modules`, and `onescience.utils` are intentionally imported from the installed OneScience environment.

