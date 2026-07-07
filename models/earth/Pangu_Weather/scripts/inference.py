import torch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "model"))

import glob
import numpy as np
import h5py
from tqdm import tqdm
from pangu_weather.models import Pangu
from pangu_weather.data import ERA5Datapipe
from pangu_weather.utils import YParams


def get_stats(data_dir, channels):
    """从新版 h5 中读取变量列表与归一化参数（均值/标准差）"""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]
        mu = f["global_means"][:]   # [1, C, 1, 1]
        std = f["global_stds"][:]

    channel_indices = [all_variables.index(v) for v in channels]
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


if __name__ == "__main__":
    current_path = PROJECT_ROOT
    os.chdir(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "config/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)

    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
        num_workers=4,
    )
    test_dataloader, _ = datapipe.get_dataloader("test")

    static_dir = os.path.join(cfg_data.dataset.data_dir, "static")
    
    land_mask = torch.from_numpy(np.load(os.path.join(static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to('cuda:0')
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)

    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location="cuda:0")
    model = Pangu(img_size=cfg_data.dataset.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  num_heads=cfg.num_heads,
                  window_size=cfg.window_size,
                  ).to('cuda:0')
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    os.makedirs('result/output/', exist_ok=True)
    print(f"📂 samples will be generated to './result/output/'")
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0]
            outvar = data[1]
            filename = data[4][-1][0]
            invar_surface = invar[:, :4, :, :].to("cuda:0", dtype=torch.float32)
            invar_upper_air = invar[:, 4:, :, :].to("cuda:0", dtype=torch.float32)
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

            out_surface, out_upper_air = model(invar)
            out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
            pred_var = torch.concat([out_surface, out_upper_air], dim=1).cpu().numpy()
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)
