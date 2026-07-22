#!/usr/bin/env python3
"""GraphCast_small JAX end-to-end inference test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def reexec_in_venv() -> None:
    """Use the same virtual environment as run.sh when available."""
    venv = Path(
        os.environ.get("WEATHER_VENV", str(Path.home() / ".venvs/weather"))
    )
    python = venv / "bin" / "python"

    if not python.is_file():
        return
    if Path(sys.executable).resolve() == python.resolve():
        return
    if os.environ.get("GRAPHCAST_TEST_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["GRAPHCAST_TEST_REEXEC"] = "1"
    os.execve(
        str(python),
        [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and validate GraphCast_small JAX inference."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path(os.environ["WEATHER_DATA_ROOT"])
            if "WEATHER_DATA_ROOT" in os.environ
            else None
        ),
        help="Directory containing the assets directory.",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Optional directory in which test results are retained.",
    )
    args = parser.parse_args()

    if args.data_root is None:
        parser.error(
            "Set WEATHER_DATA_ROOT or provide --data-root."
        )
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    model_dir = Path(__file__).resolve().parent
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir
        else (
            args.data_root.resolve()
            / "test_runs"
            / f"graphcast_small_{timestamp}_{os.getpid()}"
        )
    )
    output_dir = work_dir / "outputs"
    logs_dir = work_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = work_dir / "test_summary.json"
    combined_log = work_dir / "combined.log"
    started = time.monotonic()

    summary: dict[str, object] = {
        "test": "GraphCast_small JAX end-to-end inference",
        "status": "FAIL",
        "data_root": str(args.data_root.resolve()),
        "work_dir": str(work_dir),
        "steps": args.steps,
    }

    try:
        import numpy as np
        import xarray as xr

        required = [
            model_dir / "run.sh",
            model_dir / "run_inference.py",
            model_dir / "environment_check.py",
            model_dir / "download_assets.py",
            model_dir / "requirements.txt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required files: " + ", ".join(missing)
            )

        subprocess.run(
            ["bash", "-n", str(model_dir / "run.sh")],
            check=True,
        )

        python_files = [
            model_dir / "test.py",
            model_dir / "run_inference.py",
            model_dir / "environment_check.py",
            model_dir / "download_assets.py",
            *sorted((model_dir / "graphcast").glob("*.py")),
        ]
        for path in python_files:
            compile(path.read_bytes(), str(path), "exec")

        env = os.environ.copy()
        env["WEATHER_DATA_ROOT"] = str(args.data_root.resolve())
        env["WEATHER_OUTPUT_DIR"] = str(output_dir)
        env["WEATHER_LOGS_DIR"] = str(logs_dir)
        env["WEATHER_STEPS"] = str(args.steps)

        print("Running GraphCast_small inference...")
        print(f"Test results: {work_dir}")

        with combined_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                ["bash", str(model_dir / "run.sh")],
                cwd=model_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None

            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)

            return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"run.sh exited with status {return_code}"
            )

        predictions_path = output_dir / "predictions.nc"
        metrics_path = output_dir / "metrics.json"

        for path in (predictions_path, metrics_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty output: {path}")

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        if not isinstance(metrics, dict):
            raise RuntimeError("metrics.json must contain a JSON object")

        nonfinite_count = 0
        numeric_value_count = 0

        with xr.open_dataset(predictions_path) as dataset:
            variables = sorted(dataset.data_vars)
            dimensions = {
                name: int(size) for name, size in dataset.sizes.items()
            }

            if not variables:
                raise RuntimeError(
                    "predictions.nc contains no prediction variables"
                )

            for name in variables:
                values = np.asarray(dataset[name].values)
                if np.issubdtype(values.dtype, np.number):
                    numeric_value_count += int(values.size)
                    nonfinite_count += int(
                        values.size - np.isfinite(values).sum()
                    )

        if numeric_value_count == 0:
            raise RuntimeError("No numeric prediction values were found")
        if nonfinite_count != 0:
            raise RuntimeError(
                f"Found {nonfinite_count} non-finite prediction values"
            )

        summary.update(
            {
                "status": "PASS",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "predictions_netcdf": str(predictions_path),
                "metrics_json": str(metrics_path),
                "prediction_variables": variables,
                "prediction_dimensions": dimensions,
                "numeric_value_count": numeric_value_count,
                "nonfinite_prediction_count": nonfinite_count,
                "combined_log": str(combined_log),
            }
        )

        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nTEST PASSED")
        print(f"Summary: {summary_path}")
        print(f"Predictions: {predictions_path}")
        print(f"Metrics: {metrics_path}")
        return 0

    except Exception as error:
        summary.update(
            {
                "status": "FAIL",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(error).__name__}: {error}",
                "combined_log": str(combined_log),
            }
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\nTEST FAILED: {error}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    reexec_in_venv()
    raise SystemExit(main())