"""Download pre-trained INFINITY model weights from ModelScope."""

import argparse
import os
from modelscope.hub.snapshot_download import snapshot_download


def download_weights(local_dir: str = "weight"):
    """Download model weights from OneScience/INFINITY on ModelScope."""
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading INFINITY weights from OneScience/INFINITY to {local_dir}/ ...")
    snapshot_download("OneScience/INFINITY", local_dir=local_dir, cache_dir=None)
    print("Done. Expected structure:")
    print("  weight/")
    print("  ├── outputs_tiny/")
    print("  ├── outputs_medium/")
    print("  ├── outputs_large/")
    print("  ├── outputs_large_bias/")
    print("  ├── outputs_large_n4096/")
    print("  └── outputs_smoke/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download INFINITY pre-trained weights")
    parser.add_argument("--local_dir", default="weight", help="Target directory for weights")
    args = parser.parse_args()
    download_weights(args.local_dir)
