#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility preflight entry for OneScience/OpenFold.")
    parser.add_argument("--repo-root", "--root", default=".", dest="repo_root", help="OpenFold package root")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root / "tools"))
    from preflight_openfold import main as openfold_main

    return openfold_main(["--root", str(root)])


if __name__ == "__main__":
    raise SystemExit(main())
