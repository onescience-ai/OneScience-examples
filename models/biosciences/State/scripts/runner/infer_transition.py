"""Run State Transition inference on a new AnnData file."""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from models.cli._tx._infer import add_arguments_infer, run_tx_infer


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments_infer(parser)
    run_tx_infer(parser.parse_args())


if __name__ == "__main__":
    main()
