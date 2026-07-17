import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import HeteroGNS
from onescience.datapipes.cfd import BENODatapipe
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.utils.beno.util import record_data, to_np_array
from onescience.utils.beno.utilities import LpLoss, plot_data


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("BENO_Inference")


def set_default_data_env():
    os.environ.setdefault("ONESCIENCE_BENO_DATA_DIR", str(PROJECT_ROOT / "data"))


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config():
    set_default_data_env()
    cfg = YParams(str(PROJECT_ROOT / "conf" / "config.yaml"), "root")
    cfg.datapipe.source.data_dir = str(resolve_path(cfg.datapipe.source.data_dir))
    cfg.datapipe.source.cache_dir = str(resolve_path(cfg.datapipe.source.cache_dir))
    cfg.inference.checkpoint_dir = str(resolve_path(cfg.inference.checkpoint_dir))
    cfg.inference.result_dir = str(resolve_path(cfg.inference.result_dir))
    cfg.inference.picture_dir = str(resolve_path(cfg.inference.picture_dir))
    cfg.inference.metrics_path = str(resolve_path(cfg.inference.metrics_path))
    if cfg.inference.checkpoint_path:
        cfg.inference.checkpoint_path = str(resolve_path(cfg.inference.checkpoint_path))
    return cfg


def activation_from_name(name):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "leakyrelu":
        return nn.LeakyReLU
    return nn.SiLU


def build_model(model_cfg):
    return HeteroGNS(
        nnode_in_features=model_cfg.nnode_in_features,
        nnode_out_features=model_cfg.nnode_out_features,
        nedge_in_features=model_cfg.nedge_in_features,
        latent_dim=model_cfg.get("latent_dim", model_cfg.get("width", 128)),
        nmessage_passing_steps=model_cfg.get("nmessage_passing_steps", 10),
        nmlp_layers=model_cfg.nmlp_layers,
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", model_cfg.get("width", 128)),
        activation=activation_from_name(model_cfg.act),
        boundary_dim=model_cfg.boundary_dim,
        trans_layer=model_cfg.trans_layer,
    )


def select_device(device_name, dist):
    device_name = str(device_name).lower()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("inference.device is cuda, but CUDA is not available.")
        return torch.device(f"cuda:{dist.local_rank}")
    return dist.device


def find_checkpoint(cfg):
    if cfg.inference.checkpoint_path:
        path = Path(cfg.inference.checkpoint_path)
        return path if path.exists() else None
    checkpoints = sorted(Path(cfg.inference.checkpoint_dir).glob("model_epoch_*.pt"))
    return checkpoints[-1] if checkpoints else None


def main():
    cfg = load_config()
    DistributedManager.initialize()
    dist = DistributedManager()
    device = select_device(cfg.inference.get("device", "auto"), dist)

    datapipe = BENODatapipe(cfg, distributed=(dist.world_size > 1))
    test_loader, _ = datapipe.test_dataloader()
    u_normalizer = datapipe.u_normalizer.to(device)
    a_normalizer = datapipe.a_normalizer.to(device)
    resolution = int(cfg.datapipe.data.resolution)

    model = build_model(cfg.model).to(device)
    checkpoint = find_checkpoint(cfg)
    if checkpoint is not None:
        LOGGER.info("Loading checkpoint: %s", checkpoint)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        LOGGER.warning("No checkpoint found. Running with random weights.")
    model.eval()

    result_dir = Path(cfg.inference.result_dir)
    picture_dir = Path(cfg.inference.picture_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    picture_dir.mkdir(parents=True, exist_ok=True)

    myloss = LpLoss(size_average=False)
    analysis_record = {}
    out_all = np.array([])
    label_all = np.array([])
    a_ori_all = np.array([])
    mask_all = np.array([])
    grid_all = np.array([])

    with torch.no_grad():
        for sample_id, data in enumerate(test_loader):
            data = data.to(device)
            out_indomain = model(data)

            full_out = torch.zeros((resolution * resolution, 1), device=device)
            full_label = torch.zeros((resolution * resolution), device=device)
            full_input_a = torch.zeros((resolution * resolution, 10), device=device)
            full_grid = torch.zeros((resolution * resolution, 2), device=device)
            indices = data["G1"].sample_idx

            full_out[indices] = out_indomain
            full_label[indices] = data["G1+2"].y
            full_input_a[indices, :] = data["G1"].x
            full_grid[indices, :] = data["G1"].x[:, :2]

            pred_decoded = u_normalizer.decode(full_out.view(1, -1))
            a_decoded = a_normalizer.decode(full_input_a[:, 2].view(1, -1))
            label_reshaped = full_label.view(1, -1)

            cell_state_full = torch.zeros((1, resolution * resolution), device=device)
            cell_state_full[0, :] = data["G1"].cell_state

            l2_item = myloss(pred_decoded, label_reshaped).item()
            mae_item = nn.L1Loss()(pred_decoded, label_reshaped).item()
            record_data(analysis_record, [l2_item, mae_item], ["L2", "MAE"])

            np.savez(
                result_dir / f"sample_{sample_id:04d}.npz",
                predict=to_np_array(pred_decoded).reshape(resolution, resolution),
                label=to_np_array(label_reshaped).reshape(resolution, resolution),
                forcing=to_np_array(a_decoded).reshape(resolution, resolution),
                mask=to_np_array(cell_state_full).reshape(resolution, resolution),
            )

            out_all = np.append(out_all, to_np_array(pred_decoded))
            label_all = np.append(label_all, to_np_array(label_reshaped))
            a_ori_all = np.append(a_ori_all, to_np_array(a_decoded))
            mask_all = np.append(mask_all, to_np_array(cell_state_full))
            grid_all = np.append(grid_all, to_np_array(full_grid.unsqueeze(0)))

    metrics = {
        "mean_l2": float(np.mean(analysis_record["L2"])),
        "std_l2": float(np.std(analysis_record["L2"])),
        "mean_mae": float(np.mean(analysis_record["MAE"])),
        "num_samples": int(len(analysis_record["L2"])),
    }
    metrics_path = Path(cfg.inference.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Metrics: %s", metrics)

    plot_samples = min(int(cfg.inference.num_visualize), metrics["num_samples"])
    if plot_samples >= 2:
        plot_path = picture_dir / "forcing_solution_comparison.png"
        plot_data(
            predict_term=out_all,
            true_term=label_all,
            forcing_term=a_ori_all,
            forcing_mask=mask_all,
            grid_info=grid_all,
            resolution=resolution,
            num_samples=plot_samples,
            interpolation="bilinear",
            save_path=str(plot_path),
        )
        LOGGER.info("Plot saved to %s", plot_path)
    elif plot_samples == 1:
        LOGGER.info("Skipping plot_data for one sample because the runtime helper expects a 2D axes grid.")

    dist.cleanup()


if __name__ == "__main__":
    main()
