#!/usr/bin/env python3
"""ProteinMPNN-specific compatibility entry for the standard preflight."""
from pathlib import Path
import runpy
import sys

script = Path(__file__).with_name("preflight_check.py")
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
