#!/usr/bin/env python3
"""Local end-to-end smoke test for openclimatefix-models/dgmr."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "pytorch_lightning": "pytorch-lightning",
    "einops": "einops",
    "antialiased_cnns": "antialiased-cnns",
    "pytorch_msssim": "pytorch-msssim",
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the local DGMR checkpoint and run one inference pass."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "skillful_nowcasting",
        help="Local openclimatefix/skillful_nowcasting checkout.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "pretrained" / "dgmr",
        help="Directory containing config.json and pytorch_model.bin.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "test_results" / "dgmr_smoke",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow auto mode to fall back to slow CPU inference.",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=1,
        help="Use 1 for a smoke test and 18 for the full checkpoint horizon.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check files and imports without loading the checkpoint.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def select_device(torch, requested: str, allow_cpu: bool) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False."
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise RuntimeError(
        "torch.cuda.is_available() is False. This normally means that the "
        "container's PyTorch build and the accelerator driver/DTK do not match. "
        "Use --device cpu --allow-cpu only for a slow CPU diagnostic."
    )


def load_state_dict(torch, checkpoint: Path) -> dict:
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    state = torch.load(checkpoint, **options)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise TypeError("Checkpoint does not contain a non-empty state_dict.")
    if all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def main() -> int:
    args = arguments()
    if args.forecast_steps < 1:
        raise SystemExit("--forecast-steps must be at least 1.")

    source_dir = args.source_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    report_path = work_dir / "report.json"
    report = {
        "model": "openclimatefix-models/dgmr",
        "status": "RUNNING",
        "stage": "startup",
        "started_at": now(),
        "command": sys.argv,
        "paths": {
            "source_dir": str(source_dir),
            "model_dir": str(model_dir),
            "work_dir": str(work_dir),
        },
    }
    save_report(report_path, report)

    try:
        report["stage"] = "file_check"
        config_path = model_dir / "config.json"
        checkpoint = model_dir / "pytorch_model.bin"
        required_files = (
            source_dir / "dgmr" / "__init__.py",
            source_dir / "dgmr" / "dgmr.py",
            config_path,
            checkpoint,
        )
        missing_files = [str(path) for path in required_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "Missing required files:\n- " + "\n- ".join(missing_files)
            )
        report["files"] = {
            "checkpoint_bytes": checkpoint.stat().st_size,
            "config_bytes": config_path.stat().st_size,
        }
        save_report(report_path, report)

        report["stage"] = "runtime_import"
        missing_packages = [
            distribution
            for module, distribution in REQUIRED_IMPORTS.items()
            if importlib.util.find_spec(module) is None
        ]
        if missing_packages:
            raise RuntimeError(
                "Missing Python packages: "
                + ", ".join(missing_packages)
                + ". Do not install or upgrade torch/torchvision automatically."
            )

        torch = importlib.import_module("torch")
        importlib.import_module("torchvision")
        importlib.import_module("pytorch_lightning")
        sys.path.insert(0, str(source_dir))
        DGMR = importlib.import_module("dgmr").DGMR

        cuda_available = bool(torch.cuda.is_available())
        report["environment"] = {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                distribution: version(distribution)
                for distribution in (
                    "torch",
                    "torchvision",
                    "pytorch-lightning",
                    "torchmetrics",
                    "einops",
                    "antialiased-cnns",
                    "pytorch-msssim",
                    "huggingface-hub",
                    "safetensors",
                    "numpy",
                )
            },
            "accelerator": {
                "cuda_available": cuda_available,
                "cuda_version": getattr(torch.version, "cuda", None),
                "hip_version": getattr(torch.version, "hip", None),
                "device_count": int(torch.cuda.device_count())
                if cuda_available
                else 0,
                "devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ]
                if cuda_available
                else [],
            },
        }
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report["checkpoint_config"] = config
        save_report(report_path, report)

        if args.check_only:
            report.update(
                status="CHECK_ONLY_PASS", stage="complete", finished_at=now()
            )
            save_report(report_path, report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        device = select_device(torch, args.device, args.allow_cpu)
        report.update(stage="checkpoint_load", device=device)
        save_report(report_path, report)

        smoke_config = dict(config)
        smoke_config.update(forecast_steps=args.forecast_steps, num_samples=1)
        model = DGMR(**smoke_config)
        state = load_state_dict(torch, checkpoint)
        model.load_state_dict(state, strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        generator = model.generator.eval()
        del model, state
        gc.collect()
        generator = generator.to(device)

        report["stage"] = "inference"
        save_report(report_path, report)
        torch.manual_seed(0)
        if device == "cuda":
            torch.cuda.manual_seed_all(0)
            torch.cuda.reset_peak_memory_stats()

        height = int(config["output_shape"])
        channels = int(config["input_channels"])
        model_input = torch.rand(
            (1, 4, channels, height, height),
            dtype=torch.float32,
            device=device,
        )
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = generator(model_input)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        expected = (1, args.forecast_steps, channels, height, height)
        actual = tuple(int(value) for value in prediction.shape)
        if actual != expected:
            raise AssertionError(f"Output shape {actual}; expected {expected}.")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Prediction contains NaN or Inf values.")

        output = prediction.detach().float().cpu()
        prediction_path = work_dir / "prediction.pt"
        torch.save(output, prediction_path)
        report.update(
            {
                "status": "PASS",
                "stage": "complete",
                "finished_at": now(),
                "inference": {
                    "parameter_count": int(parameter_count),
                    "input_shape": list(model_input.shape),
                    "output_shape": list(actual),
                    "elapsed_seconds": elapsed,
                    "output_min": float(output.min()),
                    "output_max": float(output.max()),
                    "output_mean": float(output.mean()),
                    "output_is_finite": True,
                    "peak_accelerator_memory_bytes": int(
                        torch.cuda.max_memory_allocated()
                    )
                    if device == "cuda"
                    else None,
                    "prediction_file": str(prediction_path),
                },
            }
        )
        save_report(report_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "FAIL",
                "finished_at": now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        save_report(report_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())