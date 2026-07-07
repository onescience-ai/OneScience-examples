from pathlib import Path
import os
import sys


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def prepare_runtime() -> Path:
    root = package_root()
    model_dir = root / "model"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    os.chdir(root)
    return root
