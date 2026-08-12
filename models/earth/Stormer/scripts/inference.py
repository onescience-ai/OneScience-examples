"""
Stormer Inference Script — Matching Official Implementation.

Performs autoregressive weather forecasting following the official
forward_validation logic exactly:

    For each step:
        norm_diff = model(x_norm, interval)        # predict in diff-normalized space
        raw_diff = reverse_diff_transform(norm_diff)  # → original space
        pred_raw = reverse_inp_transform(x_norm) + raw_diff  # → original value
        x_norm = inp_transform(pred_raw)             # re-normalize for next step

Each target lead time uses all compatible base intervals [6, 12, 24],
then ensemble-averages the predictions.

Usage:
    python scripts/inference.py
"""

import torch
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings from external libraries
warnings.filterwarnings("ignore", category=UserWarning, module="apex")
warnings.filterwarnings("ignore", message=".*DtypeTensor constructors.*")

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import glob
import numpy as np
import h5py
from tqdm import tqdm
from model.stormer import Stormer, CONSTANTS
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe


# ============================================================================
# Normalization utilities (same as train.py)
# ============================================================================

class Normalize:
    """Per-variable normalization: y = (x - mean) / std."""

    def __init__(self, mean, std, device='cpu'):
        self.mean = mean.view(1, -1, 1, 1).to(device)
        self.std = std.view(1, -1, 1, 1).to(device)

    def __call__(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
            return ((x - self.mean) / self.std).squeeze(0)
        return (x - self.mean) / self.std


def get_reverse_transform(transform):
    """Return the inverse of a Normalize transform."""
    mean = transform.mean.view(-1)
    std = transform.std.view(-1)
    std_rev = 1.0 / std
    mean_rev = -mean * std_rev
    return Normalize(mean_rev, std_rev, device=transform.mean.device)


def load_normalization_stats(normalize_dir, variables, device):
    """Load official Stormer normalization constants."""
    # Input normalization
    mean_dict = dict(np.load(os.path.join(normalize_dir, "normalize_mean.npz")))
    std_dict = dict(np.load(os.path.join(normalize_dir, "normalize_std.npz")))

    inp_mean = np.concatenate([mean_dict[v] for v in variables], axis=0)
    inp_std = np.concatenate([std_dict[v] for v in variables], axis=0)

    inp_mean_t = torch.from_numpy(inp_mean).float()
    inp_std_t = torch.from_numpy(inp_std).float()

    inp_transform = Normalize(inp_mean_t, inp_std_t, device)
    reverse_inp_transform = get_reverse_transform(inp_transform)

    # Diff normalization for each interval
    reverse_diff_transform = {}
    for interval in [6, 12, 24]:
        dmean_dict = dict(np.load(
            os.path.join(normalize_dir, f"normalize_diff_mean_{interval}.npz")))
        dstd_dict = dict(np.load(
            os.path.join(normalize_dir, f"normalize_diff_std_{interval}.npz")))

        dmean = np.concatenate([dmean_dict[v] for v in variables], axis=0)
        dstd = np.concatenate([dstd_dict[v] for v in variables], axis=0)

        dmean_t = torch.from_numpy(dmean).float()
        dstd_t = torch.from_numpy(dstd).float()

        diff_transform = Normalize(dmean_t, dstd_t, device)
        reverse_diff_transform[interval] = get_reverse_transform(diff_transform)

    return inp_transform, reverse_inp_transform, reverse_diff_transform


def _replace_constant(yhat, out_variables):
    """Zero out diffs for constant/invariant variables."""
    for i in range(yhat.shape[1]):
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = 0.0
    return yhat


def get_stats(data_dir, channels):
    """Read normalization statistics from HDF5 (for denormalizing output)."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [
            v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]
        ]
        mu = f["global_means"][:]
        std = f["global_stds"][:]

    channel_indices = [all_variables.index(v) for v in channels]
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


# ============================================================================
# Inference
# ============================================================================

def autoregressive_rollout(model, x, variables, interval, steps, device,
                           inp_transform, reverse_inp_transform,
                           reverse_diff_transform):
    """Autoregressive rollout matching official forward_validation.

    Args:
        model: Stormer model
        x: (1, V, H, W) initial state in INPUT-NORMALIZED space
        variables: list of variable names
        interval: base interval in hours
        steps: number of autoregressive steps
        device: torch device

    Returns:
        x: (1, V, H, W) final predicted state in INPUT-NORMALIZED space
    """
    interval_tensor = torch.tensor([interval], device=device, dtype=torch.float32)

    for _ in range(steps):
        # Predict diff in diff-normalized space
        norm_diff = model(x, variables, interval_tensor)
        norm_diff = _replace_constant(norm_diff, variables)

        # Convert diff from diff-normalized → original space
        raw_diff = reverse_diff_transform[interval](norm_diff)

        # Convert input from input-normalized → original space
        pred_raw = reverse_inp_transform(x) + raw_diff

        # Re-normalize for next step
        x = inp_transform(pred_raw)

    return x


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    # Config
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    variables = cfg_data.dataset.channels

    # Load normalization stats
    normalize_dir = cfg.normalize_dir
    (inp_transform, reverse_inp_transform,
     reverse_diff_transform) = load_normalization_stats(
         normalize_dir, variables, device)
    print(f"✅ Normalization stats loaded from {normalize_dir}")

    # Load HDF5 stats for final output denormalization
    means, stds = get_stats(cfg_data.dataset.data_dir, variables)

    # DataLoader for test set (raw single steps)
    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=variables,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
        num_workers=4,
        input_steps=1,
        output_steps=1,
        normalize=False,  # Raw data — we apply official normalization
    )
    test_dataloader, _ = datapipe.get_dataloader("test")

    # Load model
    ckpt_path = f"{cfg.checkpoint_dir}/model_bak.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"❌ Checkpoint not found at {ckpt_path}. "
            "Please train the model first."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Stormer(
        in_img_size=cfg.in_img_size,
        variables=variables,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Model loaded from {ckpt_path}")

    list_intervals = cfg.list_train_intervals
    val_lead_times = cfg.val_lead_times

    os.makedirs('result/output/', exist_ok=True)
    print(f"\n📂 Predictions will be saved to './result/output/'")

    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring test set", unit="batch"):
            invar = data[0].to(device, dtype=torch.float32).squeeze(0)  # (C, H, W) raw
            filename = data[4][-1][0]  # time_index

            # Normalize input with official stats
            x_norm = inp_transform(invar).unsqueeze(0)  # (1, V, H, W)

            for lead_time in val_lead_times:
                all_preds = []

                for interval in list_intervals:
                    if lead_time % interval == 0:
                        steps = lead_time // interval
                        pred_norm = autoregressive_rollout(
                            model, x_norm, variables, interval, steps, device,
                            inp_transform, reverse_inp_transform,
                            reverse_diff_transform,
                        )
                        all_preds.append(pred_norm)

                if all_preds:
                    ensemble_pred_norm = torch.stack(all_preds, dim=0).mean(0)
                else:
                    interval = list_intervals[0]
                    steps = lead_time // interval
                    ensemble_pred_norm = autoregressive_rollout(
                        model, x_norm, variables, interval, steps, device,
                        inp_transform, reverse_inp_transform,
                        reverse_diff_transform,
                    )

                # Denormalize: pred_raw = reverse_inp(pred_norm)
                pred_raw = reverse_inp_transform(ensemble_pred_norm).cpu().numpy()

                # Save
                save_name = f"{filename}_lead{lead_time}h"
                np.save(f"result/output/{save_name}.npy", pred_raw)

    print(f"✅ Inference complete. Results saved to './result/output/'")
