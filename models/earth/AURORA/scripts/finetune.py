"""Fine-tune Aurora from an official or project-local checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import main  # noqa: E402


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--mode" in argv:
        mode_index = argv.index("--mode")
        if mode_index + 1 >= len(argv) or argv[mode_index + 1] != "finetune":
            raise SystemExit("scripts/finetune.py only supports --mode finetune")
    else:
        argv.extend(["--mode", "finetune"])
    raise SystemExit(main(argv))
