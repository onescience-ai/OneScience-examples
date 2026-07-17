"""Download pre-trained GeoMPNN model weights from ModelScope."""

import argparse
import os
from modelscope.hub.snapshot_download import snapshot_download


def download_weights(local_dir: str = "weight"):
    """Download model weights from OneScience/GeoMPNN on ModelScope."""
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading GeoMPNN weights from OneScience/GeoMPNN to {local_dir}/ ...")
    snapshot_download("OneScience/GeoMPNN", local_dir=local_dir, cache_dir=None)
    print("Done. Expected files:")
    print("  weight/ux_seed0.pt")
    print("  weight/uy_seed0.pt")
    print("  weight/p_seed0.pt")
    print("  weight/nut_seed0.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GeoMPNN pre-trained weights")
    parser.add_argument("--local_dir", default="weight", help="Target directory for weights")
    args = parser.parse_args()
    download_weights(args.local_dir)
