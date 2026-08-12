import torch
import os
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
import glob
import numpy as np
import h5py
from tqdm import tqdm
from model.ClimaX import ClimaX
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe


def get_stats(data_dir, channels):
    """Read variable list and normalization params (mean/std) from h5 file."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [
            v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]
        ]
        mu = f["global_means"][:]    # [1, C, 1, 1]
        std = f["global_stds"][:]

    channel_indices = [all_variables.index(v) for v in channels]
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")

    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    all_vars = cfg_data.dataset.channels
    out_vars = cfg_data.dataset.out_variables

    means, stds = get_stats(cfg_data.dataset.data_dir, out_vars)

    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=all_vars,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
        num_workers=4,
    )
    test_dataloader, _ = datapipe.get_dataloader("test")

    ckpt = torch.load(
        f"{cfg.checkpoint_dir}/model_bak.pth",
        map_location="cuda:0",
        weights_only=False,
    )
    model = ClimaX(
        default_vars=all_vars,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        decoder_depth=cfg.decoder_depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        drop_path=cfg.drop_path,
        drop_rate=cfg.drop_rate,
    ).to('cuda:0')
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    os.makedirs('result/output/', exist_ok=True)
    print(f"Saving predictions to './result/output/'")

    predict_range = cfg.predict_range
    hrs_each_step = cfg.hrs_each_step
    lead_time_val = (predict_range * hrs_each_step) / 100.0

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to("cuda:0", dtype=torch.float32)
            filename = data[4][-1][0]  # time string from dataloader (input timestamp)

            preds = model(invar, all_vars, out_vars, lead_time_val)
            pred_var = preds.cpu().numpy()

            # Denormalize
            pred_var = pred_var * stds + means
            np.save(f"result/output/{filename}.npy", pred_var)

    print("Inference complete.")
