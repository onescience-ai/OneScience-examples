"""Evaluate State Embeddings on a perturbation AnnData file."""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from models.cli._emb._eval import add_arguments_eval, run_emb_eval


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_eval(parser)
    run_emb_eval(parser.parse_args())


if __name__ == "__main__":
    main()
