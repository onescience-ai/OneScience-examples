from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "model"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, MODEL_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import numpy as np
import torch
import yaml

from stormer import StormCast, edm_heun_sample
from data_loader import StormCastDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StormCast inference")
    parser.add_argument("--config", type=Path, default=Path("conf/config.yaml"))
    parser.add_argument("--regression-weights", type=Path)
    parser.add_argument("--diffusion-weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--diffusion-steps", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _resolve_paths(config, config_path.parent.parent)
    inference_config = config["inference"]
    regression_weights = args.regression_weights or Path(
        config["model"]["regression_weights"]
    )
    diffusion_weights = args.diffusion_weights or Path(
        config["model"]["diffusion_weights"]
    )
    output = args.output or Path(inference_config["output_dir"]) / "forecast.h5"
    run_inference(
        config=config,
        regression_weights=regression_weights,
        diffusion_weights=diffusion_weights,
        output=output,
        num_steps=args.num_steps or inference_config["num_steps"],
        diffusion_steps=args.diffusion_steps or inference_config["diffusion_steps"],
        seed=config["project"]["seed"] if args.seed is None else args.seed,
    )


@torch.no_grad()
def run_inference(
    config: dict[str, Any],
    regression_weights: Path,
    diffusion_weights: Path,
    output: Path,
    num_steps: int,
    diffusion_steps: int,
    seed: int,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("StormCast inference requires a CUDA/HIP device")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    data_config = config["data"]
    inference_config = config["inference"]
    split = inference_config["split"]
    years_key = f"{split}_years"
    if years_key not in data_config:
        raise ValueError(f"Unknown inference split: {split}")
    dataset = StormCastDataset(
        data_root=data_config["root_dir"],
        years=data_config[years_key],
        era5_variables=data_config["era5_variables"],
        state_variables=data_config["state_variables"],
        invariant_variables=data_config["invariant_variables"],
        image_size=data_config["image_size"],
        input_steps=data_config["input_steps"],
        output_steps=data_config["output_steps"],
        normalize=data_config["normalize"],
    )
    if num_steps > len(dataset):
        raise ValueError(f"Requested {num_steps} steps but split contains {len(dataset)}")

    regression = _load_weights(regression_weights, "regression").to(device).eval()
    diffusion = _load_weights(diffusion_weights, "diffusion").to(device).eval()
    model = StormCast(regression, diffusion)
    generator = torch.Generator(device=device).manual_seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    first = dataset[0]
    state = first["state"][0].unsqueeze(0).to(device, dtype=torch.float32)
    invariant = first["invariant"].to(device, dtype=torch.float32)
    state_channels = len(data_config["state_variables"])
    background_channels = len(data_config["era5_variables"])
    height, width = data_config["image_size"]
    if [height, width] != list(config["model"]["image_size"]):
        raise ValueError("Data and model image sizes must match")
    if list(data_config["era5_image_size"]) != [721, 1440]:
        raise ValueError("ERA5 grid must be 721 x 1440")

    with h5py.File(output, "w") as handle:
        handle.attrs["normalized"] = bool(data_config["normalize"])
        handle.attrs["seed"] = seed
        handle.attrs["diffusion_steps"] = diffusion_steps
        handle.attrs["sigma_min"] = inference_config["sigma_min"]
        handle.attrs["sigma_max"] = inference_config["sigma_max"]
        handle.attrs["rho"] = inference_config["rho"]
        handle.attrs["regression_weights"] = str(regression_weights.resolve())
        handle.attrs["diffusion_weights"] = str(diffusion_weights.resolve())
        handle.attrs["state_variables"] = np.asarray(
            data_config["state_variables"], dtype="S"
        )
        handle.attrs["background_variables"] = np.asarray(
            data_config["era5_variables"], dtype="S"
        )
        prediction_store = handle.create_dataset(
            "prediction", (num_steps, state_channels, height, width), dtype="f4"
        )
        regression_store = handle.create_dataset(
            "regression", (num_steps, state_channels, height, width), dtype="f4"
        )
        target_store = handle.create_dataset(
            "target", (num_steps, state_channels, height, width), dtype="f4"
        )
        background_store = handle.create_dataset(
            "background", (num_steps, background_channels, height, width), dtype="f4"
        )
        time_store = handle.create_dataset("time_index", (num_steps,), dtype="i8")

        for index in range(num_steps):
            sample = dataset[index]
            background = sample["background"].unsqueeze(0).to(
                device, dtype=torch.float32
            )
            regression_prediction = model.predict_regression(
                state, background, invariant
            )
            condition = model.diffusion_condition(
                state, regression_prediction, invariant
            )
            residual = edm_heun_sample(
                diffusion,
                condition,
                output_channels=state_channels,
                num_steps=diffusion_steps,
                sigma_min=inference_config["sigma_min"],
                sigma_max=inference_config["sigma_max"],
                rho=inference_config["rho"],
                generator=generator,
            )
            prediction = regression_prediction + residual

            prediction_store[index] = prediction[0].cpu().numpy()
            regression_store[index] = regression_prediction[0].cpu().numpy()
            target_store[index] = sample["state"][1].numpy()
            background_store[index] = sample["background"].numpy()
            time_store[index] = int(np.asarray(sample["time_index"]).reshape(-1)[-1])
            state = prediction
            print(f"forecast_step={index + 1}/{num_steps}")
    print(f"output={output}")
    return output


def _load_weights(path: Path, kind: str) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {kind} weights: {path}. "
            "Train the corresponding stage first or pass an explicit weight path."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config")
    if saved_config is None:
        raise ValueError("Project .pt weights must include their training config")
    model = _build_stage_models_from_config(saved_config, kind)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def _build_stage_models_from_config(config: dict[str, Any], kind: str) -> torch.nn.Module:
    from stormer import build_diffusion_model, build_regression_model

    data_config = config["data"]
    model_config = config["model"]
    common = {
        "image_size": model_config["image_size"],
        "state_channels": len(data_config["state_variables"]),
        "invariant_channels": len(data_config["invariant_variables"]),
        "model_channels": model_config["model_channels"],
        "channel_mult": model_config["channel_mult"],
        "num_blocks": model_config["num_blocks"],
        "attn_resolutions": model_config["attention_resolutions"],
    }
    if kind == "regression":
        return build_regression_model(
            **common, background_channels=len(data_config["era5_variables"])
        )
    return build_diffusion_model(**common)


def _resolve_paths(config: dict[str, Any], project_root: Path) -> None:
    for section, key in (("data", "root_dir"), ("inference", "output_dir")):
        path = Path(config[section][key])
        if not path.is_absolute():
            config[section][key] = str((project_root / path).resolve())
    for key in ("regression_weights", "diffusion_weights"):
        path = Path(config["model"][key])
        if not path.is_absolute():
            config["model"][key] = str((project_root / path).resolve())


if __name__ == "__main__":
    main()
