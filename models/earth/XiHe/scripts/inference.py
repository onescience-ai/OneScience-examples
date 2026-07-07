import glob
import os

import h5py
import numpy as np
import torch

from tqdm import tqdm

from _bootstrap import prepare_runtime

current_path = str(prepare_runtime())

from xihe_src.datapipes.climate import CMEMSDatapipe
from xihe_src.models.xihe import Xihe
from xihe_src.utils import YParams


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_stats(data_dir, stats_dir, channels):
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]

    channel_indices = [all_variables.index(v) for v in channels]
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


if __name__ == "__main__":
    config_file_path = os.path.join(current_path, "config/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    device = get_device()

    means, stds = get_stats(
        cfg_data.dataset.data_dir,
        cfg_data.dataset.stats_dir,
        cfg_data.dataset.channels,
    )

    datapipe = CMEMSDatapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
        num_workers=cfg_data.dataloader.num_workers,
    )
    test_dataloader, _ = datapipe.get_dataloader("test")

    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location=device, weights_only=False)
    model = Xihe(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    os.makedirs("result/output/", exist_ok=True)
    print("samples will be generated to './result/output/'")
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to(device, dtype=torch.float32)
            filename = data[4][-1][0]

            outvar_pred = model(invar)
            pred_var = outvar_pred.cpu().numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)
