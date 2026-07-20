"""Generate State Embeddings for a new AnnData file."""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from models.cli._emb._transform import add_arguments_transform, run_emb_transform


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_transform(parser)
    run_emb_transform(parser.parse_args())


if __name__ == "__main__":
    main()
