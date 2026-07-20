"""Train a State Embedding model using a YAML configuration."""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from models.cli._emb._fit import add_arguments_fit, run_emb_fit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_fit(parser)
    args = parser.parse_args()
    from omegaconf import OmegaConf

    default_config = Path(__file__).resolve().parents[2] / "configs" / "embedding" / "state-defaults.yaml"
    config = OmegaConf.load(args.conf) if args.conf else OmegaConf.load(default_config)
    run_emb_fit(config, args)


if __name__ == "__main__":
    main()
