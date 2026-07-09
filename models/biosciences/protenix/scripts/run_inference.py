#!/usr/bin/env python3
"""Run the default Protenix inference scenario from this standalone package."""

from __future__ import annotations

import atexit
import datetime as _datetime
import os
import sys
import traceback
from pathlib import Path
from typing import TextIO


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

os.chdir(PACKAGE_ROOT)
os.environ.setdefault("DATA_ROOT_DIR", "../bio_protenix_dataset")


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def _setup_run_log() -> Path:
    log_dir = Path(os.environ.get("PROTENIX_LOG_DIR", PACKAGE_ROOT / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(os.environ.get("PROTENIX_INFERENCE_LOG", log_dir / f"run_inference_{timestamp}.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "a", buffering=1)
    atexit.register(log_file.close)

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"[run_inference] log_file={log_path}")
    print(f"[run_inference] cwd={PACKAGE_ROOT}")
    print(f"[run_inference] argv={' '.join(sys.argv)}")
    print(f"[run_inference] DATA_ROOT_DIR={os.environ.get('DATA_ROOT_DIR', '')}")
    return log_path


if __name__ == "__main__":
    log_path = _setup_run_log()
    try:
        from scripts.runner.inference_unified import main

        main()
    except BaseException:
        print(f"[run_inference] failed. See log: {log_path}", file=sys.stderr)
        traceback.print_exc()
        raise
