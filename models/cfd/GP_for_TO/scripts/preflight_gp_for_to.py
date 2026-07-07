#!/usr/bin/env python3
"""Preflight checks for the GP_for_TO ModelScope runtime package."""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import re
import sys
from pathlib import Path


PROBLEMS = ("doublepipe", "diffuser", "rugby", "pipebend")
REQUIRED_FILES = (
    "README.md",
    "onescience_run_manifest.yaml",
    "main_TO.py",
    "train.py",
    "slurm.sh",
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def read_text(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def check_required_files(repo_root: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (repo_root / name).is_file()]
    if missing:
        fail("missing required model package files: " + ", ".join(missing))
    ok("required package files are present")


def check_main_entry(repo_root: Path) -> None:
    text = read_text(repo_root / "main_TO.py")
    tree = ast.parse(text, filename=str(repo_root / "main_TO.py"))
    parser_choices = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            args = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            if "--problem" not in args:
                continue
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                    parser_choices = tuple(
                        item.value for item in keyword.value.elts if isinstance(item, ast.Constant)
                    )
    if tuple(parser_choices or ()) != PROBLEMS:
        fail(f"unexpected --problem choices: {parser_choices!r}")

    checks = {
        "N_train_per_BC = 25": r"N_train_per_BC\s*=\s*25",
        "N_col_domain = 10000": r"N_col_domain\s*=\s*10000",
        "num_iter=50000": r"num_iter\s*=\s*50000",
        "output_names": r"output_names\s*=\s*\[\s*['\"]u['\"],\s*['\"]v['\"],\s*['\"]p['\"],\s*['\"]ro['\"]\s*\]",
    }
    for label, pattern in checks.items():
        if not re.search(pattern, text):
            fail(f"main_TO.py is missing expected setting: {label}")
    ok("main_TO.py arguments and topology-optimization settings are valid")


def check_train_helpers(repo_root: Path) -> None:
    text = read_text(repo_root / "train.py")
    for name in ("find_TO", "calculate_loss_multioutput", "gamma_values", "checkpoints"):
        if name not in text:
            fail(f"train.py is missing required symbol: {name}")
    for problem in PROBLEMS:
        if problem not in text:
            fail(f"train.py does not reference problem-specific settings for {problem}")
    ok("train.py helper functions and problem settings are present")


def check_slurm(repo_root: Path) -> None:
    text = read_text(repo_root / "slurm.sh")
    if "python main_TO.py --problem doublepipe --gpu 0" not in text:
        fail("slurm.sh does not contain the expected GP_for_TO launch command")
    ok("slurm.sh launch command is present")


def maybe_add_onescience_src(repo_root: Path, explicit_src: str | None) -> None:
    candidates = []
    if explicit_src:
        candidates.append(Path(explicit_src).expanduser())
    env_src = os.environ.get("ONESCIENCE_SRC")
    if env_src:
        candidates.append(Path(env_src).expanduser())
    for parent in (repo_root, *repo_root.parents):
        candidates.append(parent / "onescience" / "src")
    for candidate in candidates:
        if (candidate / "onescience").is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            return


def check_runtime_imports(repo_root: Path, problem: str, onescience_src: str | None) -> None:
    maybe_add_onescience_src(repo_root, onescience_src)
    for package in ("torch", "gpytorch", "numpy", "matplotlib", "tqdm"):
        try:
            importlib.import_module(package)
        except Exception as exc:  # noqa: BLE001
            fail(f"runtime dependency import failed for {package}: {exc}")

    try:
        from onescience.models.GPs import GPPLUS  # noqa: F401
        from onescience.utils.GP_TO import get_data_fluid, set_seed
    except Exception as exc:  # noqa: BLE001
        fail(f"OneScience GP_for_TO imports failed: {exc}")

    set_seed(11)
    x_col, x_train, sol_train = get_data_fluid(problem=problem, N_col_domain=10000, N_train=25)
    if tuple(x_col.shape) != (10000, 2):
        fail(f"unexpected collocation shape for {problem}: {tuple(x_col.shape)}")
    if len(x_train) != 4 or len(sol_train) != 4:
        fail("get_data_fluid must return four training inputs and four solution arrays")
    if any(t.ndim != 2 or t.shape[1] != 2 for t in x_train):
        fail("boundary training coordinate arrays must have shape [n, 2]")
    if any(s.ndim != 1 for s in sol_train):
        fail("boundary solution arrays must be rank-1 tensors")
    ok(f"runtime imports and generated data schema are valid for problem={problem}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight GP_for_TO runtime package.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--problem", default="doublepipe", choices=PROBLEMS)
    parser.add_argument("--check-imports", action="store_true", help="also import torch/gpytorch/onescience and validate generated tensors")
    parser.add_argument("--onescience-src", default=None, help="optional path to onescience/src when OneScience is not installed")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.is_dir():
        fail(f"repo root does not exist: {repo_root}")

    check_required_files(repo_root)
    check_main_entry(repo_root)
    check_train_helpers(repo_root)
    check_slurm(repo_root)
    ok("no external dataset files are required; GP_for_TO generates collocation and boundary samples at runtime")
    if args.check_imports:
        check_runtime_imports(repo_root, args.problem, args.onescience_src)
    else:
        warn("runtime import checks skipped; rerun with --check-imports after installing OneScience CFD dependencies")
    ok("model preflight completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
