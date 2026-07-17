"""GeoMPNN training script.

The training code for GeoMPNN is part of the AIRS library:
https://github.com/divelab/AIRS

Usage (after installing AIRS):

    # Train a model for each field separately
    python -m airs.scripts.train_geompnn --config config/config.yaml --field ux
    python -m airs.scripts.train_geompnn --config config/config.yaml --field uy
    python -m airs.scripts.train_geompnn --config config/config.yaml --field p
    python -m airs.scripts.train_geompnn --config config/config.yaml --field nut

This script serves as a reference entry point. Refer to the AIRS library
documentation for detailed training options.
"""
import sys

if __name__ == "__main__":
    print("GeoMPNN training is implemented in the AIRS library.")
    print("See: https://github.com/divelab/AIRS")
    print()
    print("To train, install AIRS and run:")
    print("  python -m airs.scripts.train_geompnn --config config/config.yaml --field <ux|uy|p|nut>")
    sys.exit(1)
