"""Evaluate or run prediction from a trained State Transition run."""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from models.cli._tx._predict import add_arguments_predict, run_tx_predict


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_predict(parser)
    run_tx_predict(parser.parse_args())


if __name__ == "__main__":
    main()
