# Third-party resource lock

- Project: NVIDIA `torch-harmonics`
- Official repository: https://github.com/NVIDIA/torch-harmonics
- Package version: `0.8.0`
- Official tag: `v0.8.0`
- Tag commit: `c7afb5461e6c4c9298ce5afd3ada1f8436cdc15d`
- License: BSD-3-Clause
- Local installation: `.deps/`, installed with `--no-deps`; official source is not copied or modified by model code.
- Required by package metadata: Python >= 3.9, PyTorch >= 2.4.0, NumPy >= 1.22.4.

Reproducible installation from this directory:

```bash
mkdir -p .deps
python -m pip install --target .deps --no-deps -r requirements.lock
```
