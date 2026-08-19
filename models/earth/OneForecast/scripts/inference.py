"""OneForecast inference entry point with the shared ERA5 adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.era5_adapter import OFFICIAL_VARIABLES, OneForecastERA5Adapter
from model.oneforecast import build_model, check_checkpoint_compatibility, read_official_checkpoint


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["datapipe"]["dataset_dir"] = str(_resolve_path(config["datapipe"]["dataset_dir"], path))
    config["model"]["official_checkpoint_path"] = str(
        _resolve_path(config["model"]["official_checkpoint_path"], path)
    )
    config["model"]["checkpoint_path"] = config["model"]["official_checkpoint_path"]
    config["inference"]["trained_model_path"] = str(
        _resolve_path(config["inference"]["trained_model_path"], path)
    )
    config["inference"]["official_checkpoint_path"] = str(
        _resolve_path(config["inference"]["official_checkpoint_path"], path)
    )
    config["inference"]["output_dir"] = str(_resolve_path(config["inference"]["output_dir"], path))
    return config


def _resolve_device(name: str) -> torch.device:
    """Map the logical DCU name to the backend exposed by this PyTorch build."""
    requested = str(name).lower()
    if requested == "dcu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        privateuse = torch._C._get_privateuse1_backend_name()
        if privateuse != "privateuseone":
            return torch.device(privateuse)
        raise RuntimeError("runtime.device=dcu, but this PyTorch build exposes no usable accelerator")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda, but torch.cuda.is_available() is False")
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("conf/config.yaml"))
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--check-checkpoint", action="store_true")
    parser.add_argument("--model-source", choices=("trained", "official"), default=None)
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    if tuple(config["datapipe"]["variables"]) != OFFICIAL_VARIABLES:
        raise ValueError("datapipe.variables must exactly match the official 69-channel order")
    if args.model_source is not None:
        config["inference"]["model_source"] = args.model_source
    if args.check_data:
        settings = config["datapipe"]
        adapter = OneForecastERA5Adapter(
            settings["dataset_dir"], settings["test_years"], batch_size=1,
            input_steps=settings["input_steps"], output_steps=settings["output_steps"],
            normalize=settings["normalize"], num_workers=settings["num_workers"],
        )
        print(adapter.inspect())
        return
    if args.check_model:
        configured_init = config["model"].get("weight_init", "scratch")
        config["model"]["weight_init"] = "scratch"
        with __import__("torch").device("meta"):
            model = build_model(config, build_graph=False)
        print({"model": type(model).__name__, "parameters": sum(p.numel() for p in model.parameters()),
               "configured_weight_init": configured_init})
        return
    if args.check_checkpoint:
        with __import__("torch").device("meta"):
            model = build_model(config, build_graph=False)
        report = check_checkpoint_compatibility(
            model, config["model"]["official_checkpoint_path"]
        )
        print(report)
        if not report.compatible:
            raise SystemExit(1)
        return
    settings = config["datapipe"]
    if settings["input_steps"] != 1 or settings["output_steps"] != 1:
        raise SystemExit("OneForecast inference currently requires input_steps=1 and output_steps=1")
    device = _resolve_device(config["runtime"].get("device", "cpu"))
    config["model"]["weight_init"] = "scratch"
    model = build_model(config).to(device)
    source = config["inference"].get("model_source", "trained")
    checkpoint_path = config["inference"][
        "trained_model_path" if source == "trained" else "official_checkpoint_path"
    ]
    state, _ = read_official_checkpoint(checkpoint_path)
    model.load_state_dict(state)
    model.eval()
    adapter = OneForecastERA5Adapter(
        _resolve_path(settings["dataset_dir"], args.config), settings["test_years"],
        batch_size=1, input_steps=1, output_steps=1,
        normalize=settings["normalize"], num_workers=settings["num_workers"],
    )
    loader, _ = adapter.get_dataloader("test")
    output_dir = Path(config["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_batches = int(config["inference"].get("max_batches", -1))
    processed = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            inputs, targets = batch[0].float().to(device), batch[1].float().to(device)
            if inputs.ndim == 5 or targets.ndim == 5:
                raise ValueError("OneForecast currently supports input_steps=1 and output_steps=1 only")
            if inputs.ndim != 4:
                raise ValueError(f"Expected batched input with four dimensions, got {inputs.shape}")
            if inputs.shape[-2] == 121:
                inputs = inputs[..., :120, :]
            if targets.shape[-2] == 121:
                targets = targets[..., :120, :]
            if inputs.shape[-2:] != (120, 240) or targets.shape[-2:] != (120, 240):
                raise ValueError(f"Expected official model grid 120x240, got {inputs.shape} and {targets.shape}")
            prediction = model(torch.nan_to_num(inputs))
            if settings["normalize"]:
                means, stds = adapter.selected_statistics()
                prediction = prediction.cpu() * torch.from_numpy(stds).float() + torch.from_numpy(means).float()
            np.save(output_dir / f"prediction_{index:05d}.npy", prediction.cpu().numpy())
            processed += 1
            if max_batches >= 0 and index + 1 >= max_batches:
                break
    print({"output_dir": str(output_dir), "batches": processed})


if __name__ == "__main__":
    main()
