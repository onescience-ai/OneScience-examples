from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


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
from model.uno import UNO  # noqa: E402
from onescience.datapipes.cfd import NavierStokesDatapipe  # noqa: E402
from onescience.distributed.manager import DistributedManager  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def prepare_config(config: dict[str, Any]) -> None:
    datapipe = config["datapipe"]
    datapipe["source"]["data_dir"] = str(
        project_path(datapipe["source"]["data_dir"], PROJECT_ROOT).resolve()
    )


def validate_checkpoint_config(
    datapipe_config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    checkpoint_data = checkpoint["datapipe_config"]["data"]
    current_data = datapipe_config["data"]
    for key in ("t_in", "t_out", "out_dim", "downsamplex", "downsampley", "normalize"):
        if current_data[key] != checkpoint_data[key]:
            raise ValueError(
                f"Config value datapipe.data.{key}={current_data[key]} does not match "
                f"checkpoint value {checkpoint_data[key]}"
            )


def build_model(model_config: dict[str, Any], spatial_shape: tuple[int, int]) -> UNO:
    return UNO(
        in_dim=int(model_config["in_dim"]),
        out_dim=int(model_config["out_dim"]),
        spatial_shape=spatial_shape,
        hidden_dim=int(model_config["hidden_dim"]),
        modes=int(model_config["modes"]),
        space_dim=int(model_config["space_dim"]),
        include_pos=bool(model_config["include_pos"]),
        normtype=str(model_config["normtype"]),
        bilinear=bool(model_config["bilinear"]),
        activation=str(model_config["activation"]),
        pad_to_multiple=int(model_config["pad_to_multiple"]),
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
    figure, axes = plt.subplots(3, len(frames), figsize=(4 * len(frames), 9), squeeze=False)
    for row_index, (title, values) in enumerate(rows):
        for column_index, frame in enumerate(frames):
            axis = axes[row_index][column_index]
            image = axis.imshow(values[..., frame], origin="lower", cmap="viridis")
            axis.set_title(f"{title} t+{frame + 1}")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    prepare_config(config)
    training_config = config["training"]
    inference_config = config["inference"]
    datapipe_config = config["datapipe"]
    num_samples = int(inference_config["num_samples"])
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    checkpoint_path = (
        project_path(training_config["weight_dir"], PROJECT_ROOT).resolve()
        / str(training_config["checkpoint_name"])
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    validate_checkpoint_config(datapipe_config, checkpoint)
    data_file = (
        Path(datapipe_config["source"]["data_dir"])
        / datapipe_config["source"]["file_name"]
    )
    device = resolve_device(str(config["common"]["device"]))
    result_dir = project_path(
        inference_config["result_dir"], PROJECT_ROOT
    ).resolve()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_file}")
    print(f"Device: {device}")
    DistributedManager.initialize()
    try:
        datapipe = NavierStokesDatapipe(
            to_attr_dict(datapipe_config),
            distributed=False,
            normalizer_state=checkpoint.get("normalizer"),
        )
        test_loader, _ = datapipe.test_dataloader()
        spatial_shape = tuple(checkpoint["spatial_shape"])
        if tuple(datapipe.spatial_shape) != spatial_shape:
            raise ValueError(
                f"Checkpoint grid {spatial_shape} does not match data grid {datapipe.spatial_shape}"
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
            )
        prediction = datapipe.decode_solution(prediction)
        target = datapipe.decode_solution(target)
        error = relative_l2(prediction, target).mean().item()
        sample_count = min(num_samples, prediction.shape[0])
        result_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = result_dir / "prediction_sample.pt"
        image_path = result_dir / "prediction_sample.png"
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
