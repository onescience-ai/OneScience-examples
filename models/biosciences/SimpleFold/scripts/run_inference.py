#!/usr/bin/env python3
"""Run SimpleFold structure prediction from the packaged ModelScope layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import predict_structures_from_fastas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SimpleFold structure prediction.")
    parser.add_argument("--simplefold_model", default="simplefold_100M")
    parser.add_argument("--ckpt_dir", default="weight")
    parser.add_argument("--output_dir", default="outputs/minimal_inference")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--no_log_timesteps", action="store_true")
    parser.add_argument("--fasta_path", default="examples/minimal.fasta")
    parser.add_argument("--nsample_per_protein", type=int, default=1)
    parser.add_argument("--plddt", action="store_true")
    parser.add_argument("--output_format", default="mmcif", choices=["pdb", "mmcif"])
    parser.add_argument("--backend", default="torch", choices=["torch", "mlx"])
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predict_structures_from_fastas(args)


if __name__ == "__main__":
    main()

