from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LOCAL_WORKSPACE = PROJECT_ROOT.parents[1]
if (LOCAL_WORKSPACE / "onescience" / "src" / "onescience" / "datapipes").is_dir():
    sys.path.insert(0, str(LOCAL_WORKSPACE))

from common import (  # noqa: E402
    load_config,
    project_path,
    relative_l2,
    resolve_device,
    rollout,
    to_attr_dict,
)
from model.factformer import FactFormer2D  # noqa: E402
from onescience.datapipes.cfd import KolmogorovFlow2DDatapipe  # noqa: E402
from onescience.distributed.manager import DistributedManager  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def prepare_config(config: dict[str, Any]) -> None:
    datapipe = config["datapipe"]
    datapipe["source"]["data_dir"] = str(
        project_path(datapipe["source"]["data_dir"], PROJECT_ROOT).resolve()
    )
    stats_file = datapipe["data"].get("stats_file")
    if stats_file:
        datapipe["data"]["stats_file"] = str(
            project_path(stats_file, PROJECT_ROOT).resolve()
        )


def checkpoint_datapipe_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "datapipe_config" in checkpoint:
        return checkpoint["datapipe_config"]
    if "data_config" in checkpoint:
        return checkpoint["data_config"]
    raise KeyError("Checkpoint does not contain a datapipe configuration")


def validate_checkpoint_config(
    datapipe_config: dict[str, Any], checkpoint: dict[str, Any]
) -> None:
    checkpoint_data = checkpoint_datapipe_config(checkpoint)["data"]
    current_data = datapipe_config["data"]
    for key in (
        "resolution",
        "interval",
        "t_in",
        "t_out",
        "out_dim",
        "normalize",
    ):
        if current_data[key] != checkpoint_data[key]:
            raise ValueError(
                f"Config datapipe.data.{key}={current_data[key]} does not match "
                f"checkpoint value {checkpoint_data[key]}"
            )


def build_model(
    model_config: dict[str, Any], spatial_shape: tuple[int, int]
) -> FactFormer2D:
    return FactFormer2D(
        in_dim=int(model_config["in_dim"]),
        out_dim=int(model_config["out_dim"]),
        spatial_shape=spatial_shape,
        hidden_dim=int(model_config["hidden_dim"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        mlp_ratio=int(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
        activation=str(model_config["activation"]),
        include_pos=bool(model_config["include_pos"]),
        space_dim=int(model_config["space_dim"]),
        latent_multiplier=float(model_config["latent_multiplier"]),
        max_latent_steps=int(model_config["max_latent_steps"]),
    )


def save_visualization(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spatial_shape: tuple[int, int],
    output_path: Path,
) -> None:
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    height, width = spatial_shape
    frame_count = prediction.shape[-1]
    frames = sorted({0, frame_count // 2, frame_count - 1})
    sample_prediction = prediction[0].reshape(height, width, frame_count)
    sample_target = target[0].reshape(height, width, frame_count)
    rows = (
        ("Target", sample_target),
        ("Prediction", sample_prediction),
        ("Abs Error", (sample_prediction - sample_target).abs()),
    )
    figure, axes = plt.subplots(
        3, len(frames), figsize=(4 * len(frames), 9), squeeze=False
    )
    for row_index, (title, values) in enumerate(rows):
        for column_index, frame in enumerate(frames):
            axis = axes[row_index][column_index]
            image = axis.imshow(values[..., frame], origin="lower", cmap="viridis")
            axis.set_title(f"{title} t+{frame + 1}")
            axis.set_xticks([])
            axis.set_yticks([])
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.formatter = FormatStrFormatter("%.3g")
            colorbar.update_ticks()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    prepare_config(config)
    training = config["training"]
    inference = config["inference"]
    datapipe_config = config["datapipe"]
    num_samples = int(inference["num_samples"])
    if num_samples < 1:
        raise ValueError("inference.num_samples must be positive")
    checkpoint_path = (
        project_path(training["weight_dir"], PROJECT_ROOT).resolve()
        / str(training["checkpoint_name"])
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_config(datapipe_config, checkpoint)
    model_config = dict(checkpoint["model_config"])
    spatial_shape = tuple(int(value) for value in checkpoint["spatial_shape"])
    device = resolve_device(str(config["common"]["device"]))
    result_dir = project_path(inference["result_dir"], PROJECT_ROOT).resolve()

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        "Data: "
        f"{Path(datapipe_config['source']['data_dir']) / datapipe_config['source']['file_name']}"
    )
    print(f"Device: {device}")

    DistributedManager.initialize()
    try:
        datapipe = KolmogorovFlow2DDatapipe(
            to_attr_dict(datapipe_config),
            distributed=False,
            normalizer_state=checkpoint.get("normalizer"),
        )
        test_loader, _ = datapipe.test_dataloader()
        if tuple(datapipe.spatial_shape) != spatial_shape:
            raise ValueError(
                f"Checkpoint grid {spatial_shape} does not match data grid "
                f"{datapipe.spatial_shape}"
            )
        model = build_model(model_config, spatial_shape).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        batch = next(iter(test_loader))
        pos = batch["pos"].to(device)
        state = batch["x"].to(device)
        target = batch["y"].to(device)
        with torch.no_grad():
            prediction = rollout(
                model,
                pos,
                state,
                int(datapipe_config["data"]["t_out"]),
                int(datapipe_config["data"]["out_dim"]),
                int(model_config["max_latent_steps"]),
            )
        prediction = datapipe.decode_solution(prediction)
        target = datapipe.decode_solution(target)
        error = relative_l2(prediction, target).mean().item()
        sample_count = min(num_samples, prediction.shape[0])
        result_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = result_dir / str(inference["tensor_name"])
        image_path = result_dir / str(inference["figure_name"])
        torch.save(
            {
                "prediction": prediction[:sample_count].cpu(),
                "target": target[:sample_count].cpu(),
                "spatial_shape": spatial_shape,
                "relative_l2": error,
            },
            tensor_path,
        )
        save_visualization(prediction[:1], target[:1], spatial_shape, image_path)
        print(f"Relative L2: {error:.6e}")
        print(f"Saved tensor: {tensor_path}")
        print(f"Saved figure: {image_path}")
    finally:
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
