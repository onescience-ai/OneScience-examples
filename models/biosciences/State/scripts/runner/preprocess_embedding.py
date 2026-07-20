"""Create a State Embedding data profile."""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from models.cli._emb._preprocess import add_arguments_preprocess, run_emb_preprocess


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_preprocess(parser)
    run_emb_preprocess(parser.parse_args())


if __name__ == "__main__":
    main()
