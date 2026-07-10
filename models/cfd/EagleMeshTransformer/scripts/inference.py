import ctypes
import os
import sys
import sysconfig
from pathlib import Path


def preload_python_shared_library():
    """Make libpython visible to native extensions loaded with ctypes."""
    libdir = sysconfig.get_config_var("LIBDIR")
    version = sysconfig.get_config_var("VERSION")
    if not libdir or not version:
        return

    candidates = [
        Path(libdir) / f"libpython{version}.so.1.0",
        Path(libdir) / f"libpython{version}.so",
    ]
    for libpython in candidates:
        if libpython.exists():
            ctypes.CDLL(str(libpython), mode=ctypes.RTLD_GLOBAL)
            return


preload_python_shared_library()

import numpy as np
import torch
from tqdm import tqdm

# 获取项目根目录（train.py上级的上级）
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from model.graphViT import GraphViT
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.datapipes.cfd import EagleDatapipe


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else root_path / path


def fix_single_cluster_path(datapipe, cfg_data):
    if int(cfg_data.data.n_cluster) != 1:
        return

    cluster_path = Path(cfg_data.source.cluster_dir)
    for dataset_name in ("train_dataset", "val_dataset", "test_dataset"):
        dataset = getattr(datapipe, dataset_name, None)
        if dataset is not None and getattr(dataset, "cluster_path", None) is None:
            dataset.cluster_path = cluster_path


def main():
    os.chdir(root_path)
    DistributedManager.initialize()
    manager = DistributedManager()

    config_path = root_path / "config" / "config.yaml"
    cfg_model = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")
    cfg_infer = YParams(config_path, "inference")

    checkpoint_path = resolve_project_path(cfg_infer.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run `python scripts/train.py` first."
        )

    datapipe = EagleDatapipe(params=cfg_data, distributed=False)
    fix_single_cluster_path(datapipe, cfg_data)
    dataloader, _ = datapipe.test_dataloader(batch_size=int(cfg_infer.batch_size))
    device_name = cfg_infer.get("device", "auto")
    device = manager.device if device_name == "auto" else torch.device(device_name)
    model = GraphViT(state_size=cfg_model.state_size, w_size=cfg_model.w_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    output_dir = resolve_project_path(cfg_infer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    with torch.no_grad():
        for idx, x in enumerate(tqdm(dataloader, desc="Inference")):
            if not x:
                continue
            mesh_pos = x["mesh_pos"].to(device)
            edges = x["edges"].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(
                mesh_pos,
                edges,
                state,
                node_type,
                clusters,
                clusters_mask,
                apply_noise=False,
            )
            velocity_hat, pressure_hat = dataloader.dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            pred = torch.cat([velocity_hat, pressure_hat], dim=-1).cpu().numpy()
            np.save(output_dir / f"prediction_{idx:04d}.npy", pred)
            saved += 1

    print(f"Saved {saved} prediction file(s) to {output_dir}")
    manager.cleanup()


if __name__ == "__main__":
    main()
