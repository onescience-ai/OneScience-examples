#!/usr/bin/env python3
"""Collect a truthful, secret-filtered GraphCast environment report."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_IMPORTS = {
    "jax": "jax",
    "jaxlib": "jaxlib",
    "dm-haiku": "haiku",
    "numpy": "numpy",
    "xarray": "xarray",
    "torch": "torch",
    "flag-gems": "flag_gems",
    "chex": "chex",
    "dask": "dask",
    "jraph": "jraph",
    "matplotlib": "matplotlib",
    "netCDF4": "netCDF4",
    "pandas": "pandas",
    "scipy": "scipy",
    "trimesh": "trimesh",
}

SAFE_ENV_NAMES = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORM_NAME",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "WEATHER_DATA_ROOT",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)

SECRET_MARKERS = (
    "AUTH", "COOKIE", "CREDENTIAL", "KEY", "PASS", "SECRET", "TOKEN"
)


def _run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    if not shutil.which(command[0]):
        return {"available": False, "command": command[0]}
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20, cwd=cwd,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # diagnostics must continue
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}


def _version(distribution: str, module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        return {"installed": True, "version": str(version)}
    except Exception as exc:
        return {
            "installed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cpu_name() -> str:
    candidates = [platform.processor(), platform.machine()]
    if platform.system() == "Windows":
        probe = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
        ])
        candidates.insert(0, probe.get("stdout", ""))
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                candidates.insert(0, line.split(":", 1)[-1].strip())
                break
    return next((item.strip() for item in candidates if item and item.strip()), "unknown")


def _memory() -> dict[str, Any]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_bytes": mem.total,
            "total_gib": round(mem.total / 1024**3, 2),
            "available_bytes": mem.available,
            "available_gib": round(mem.available / 1024**3, 2),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _environment() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in SAFE_ENV_NAMES:
        if name not in os.environ:
            continue
        if any(marker in name.upper() for marker in SECRET_MARKERS):
            result[name] = "<redacted>"
        else:
            result[name] = os.environ[name]
    return result


def _repository(source_dir: Path) -> dict[str, Any]:
    files = [path for path in source_dir.rglob("*") if path.is_file()]
    large_files = [path for path in files if path.stat().st_size >= 10 * 1024 * 1024]
    asset_suffixes = {".nc", ".npz", ".ckpt", ".pt", ".pth", ".safetensors"}
    asset_files = [path for path in files if path.suffix.lower() in asset_suffixes]
    expected = (
        "README.md", "setup.py", "graphcast_demo.ipynb", "graphcast/graphcast.py",
        "environment_check.py", "download_assets.py", "run_inference.py", "run.sh",
    )
    git_probe = _run(["git", "rev-parse", "HEAD"], cwd=source_dir)
    commit = git_probe.get("stdout") if git_probe.get("returncode") == 0 else None
    cached_trees = sorted((source_dir / ".cache" / "huggingface" / "trees").glob("*.json"))
    cached_revision = cached_trees[0].stem if len(cached_trees) == 1 else None
    return {
        "source_url": "https://huggingface.co/Gary0205/weather/tree/main",
        "source_directory": str(source_dir),
        "current_working_directory": str(Path.cwd()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "expected_files": {
            name: (source_dir / name).is_file() for name in expected
        },
        "git_commit": commit,
        "huggingface_cached_revision": cached_revision,
        "git_probe_error": None if commit else git_probe.get("stderr", git_probe.get("error")),
        "source_fallback": None if commit else (
            "No .git metadata; use source_url and huggingface_cached_revision as provenance."
        ),
        "large_files_10_mib_or_more": [
            {"path": str(path.relative_to(source_dir)), "bytes": path.stat().st_size}
            for path in sorted(large_files)
        ],
        "model_or_netcdf_files_in_source": [
            {"path": str(path.relative_to(source_dir)), "bytes": path.stat().st_size}
            for path in sorted(asset_files)
        ],
    }


def collect(source_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "kernel": platform.uname().release,
            "machine": platform.machine(),
        },
        "hardware": {
            "cpu": _cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "memory": _memory(),
            "nvidia_smi": _run([
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]),
            "scnet_smi": _run(["ixsmi", "-L"]),
        },
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "pip": _run([sys.executable, "-m", "pip", "--version"]),
        },
        "packages": {
            name: _version(name, module)
            for name, module in PACKAGE_IMPORTS.items()
        },
        "environment": _environment(),
        "repository": _repository(source_dir),
    }

    try:
        import jax
        backend = jax.default_backend()
        devices = [str(device) for device in jax.devices()]
        non_cpu = [device for device in jax.devices() if device.platform != "cpu"]
        scnet_markers = ("scnet", "iluvatar", "mr-v100", "bi-v150")
        scnet_devices = [
            str(device) for device in non_cpu
            if any(marker in str(device).lower() for marker in scnet_markers)
        ]
        scnet_probe = report["hardware"]["scnet_smi"]
        scnet_runtime_visible = (
            scnet_probe.get("available")
            and scnet_probe.get("returncode") == 0
            and bool(scnet_probe.get("stdout"))
        )
        if scnet_devices or (non_cpu and scnet_runtime_visible):
            classification = "A: JAX recognizes an SCNET accelerator"
        elif not non_cpu:
            classification = "B: JAX recognizes CPU only"
        else:
            classification = (
                "OTHER_ACCELERATOR: JAX sees a non-CPU device, but it was not "
                "identified as an SCNET accelerator"
            )
        report["jax_runtime"] = {
            "import_ok": True,
            "default_backend": backend,
            "devices": devices,
            "classification": classification,
            "scnet_devices": scnet_devices,
        }
    except Exception as exc:
        report["jax_runtime"] = {
            "import_ok": False,
            "classification": "C: JAX import/version conflict",
            "error": f"{type(exc).__name__}: {exc}",
        }

    torch_entry: dict[str, Any]
    try:
        import torch
        torch_entry = {
            "import_ok": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except Exception as exc:
        torch_entry = {
            "import_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    report["torch_runtime"] = torch_entry
    return report


def main() -> int:
    default_root = Path(os.environ.get(
        "WEATHER_DATA_ROOT", "/root/group_data/SDU-Test/weather"
    ))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=default_root / "logs" / "environment.log"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path(__file__).resolve().parent,
        help="GraphCast source directory to audit (default: directory containing this script)",
    )
    args = parser.parse_args()
    report = collect(args.source_dir.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Environment report written to: {args.output}")
    return 0 if report["jax_runtime"]["import_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
