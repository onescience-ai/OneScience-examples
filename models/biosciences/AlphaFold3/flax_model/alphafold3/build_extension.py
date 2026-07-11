#!/usr/bin/env python3
"""Compatibility entry point for local AlphaFold3 build helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flax_model.alphafold3._build import AF3BuildError, build_all


if __name__ == "__main__":
    try:
        build_all()
    except AF3BuildError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    sys.exit(0)
