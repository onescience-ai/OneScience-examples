"""
Stormer Result Evaluation and Visualization.

Computes per-channel RMSE and ACC (Anomaly Correlation Coefficient)
for model predictions against ground truth at the corresponding lead times.

Usage:
    python scripts/result.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import glob
import h5py
from datetime import datetime, timedelta
from tqdm import tqdm
from onescience.utils.YParams import YParams
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 0.9
rcParams['xtick.major.width'] = 0.9
rcParams['ytick.major.width'] = 0.9


def get_metadata(data_dir, channels):
    """Read variable names and time_step from HDF5 attrs."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [
            v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]
        ]
        time_step = int(ds.attrs["time_step"])

    channel_indices = [all_variables.index(v) for v in channels]

    # Find all prediction files (now include lead time suffix)
    total_files = [f for f in os.listdir('./result/output/') if f.endswith('.npy')]
    total_files.sort()
    return total_files, channel_indices, time_step


def filename_to_datetime(filename_base):
    """Convert YYYYMMDDHH base filename to datetime."""
    return datetime.strptime(filename_base, "%Y%m%d%H")


def parse_pred_filename(filename):
    """Parse prediction filename like '2003010206_lead72h.npy'.

    Returns:
        base_time: datetime of input time
        lead_hours: lead time in hours
    """
    name = filename.replace('.npy', '')
    parts = name.split('_lead')
    base_str = parts[0]
    lead_str = parts[1].replace('h', '')
    base_time = datetime.strptime(base_str, "%Y%m%d%H")
    lead_hours = int(lead_str)
    return base_time, lead_hours


def get_ground_truth(data_dir, target_time, channel_indices, time_step):
    """Load ground truth from HDF5 at the target datetime.

    Args:
        data_dir: path to data directory containing data/*.h5 files
        target_time: datetime of the target time step
        channel_indices: indices of desired channels
        time_step: hours between consecutive frames

    Returns:
        label: (C, H, W) numpy array, or None if not found
    """
    year = target_time.year
    year_start = datetime(year, 1, 1)
    hours_since_year_start = (target_time - year_start).total_seconds() / 3600
    t_idx = int(hours_since_year_start / time_step)

    h5_path = os.path.join(data_dir, 'data', f'{year}.h5')
    if not os.path.exists(h5_path):
        # Try next/last year (for year boundary)
        for adj_year in [year - 1, year + 1]:
            alt_path = os.path.join(data_dir, 'data', f'{adj_year}.h5')
            if os.path.exists(alt_path):
                h5_path = alt_path
                if adj_year < year:
                    # Target is early in year, data from previous year
                    prev_start = datetime(adj_year, 1, 1)
                    year_len = int((datetime(adj_year + 1, 1, 1) - prev_start).total_seconds() / 3600)
                    t_idx = year_len // time_step + int(
                        (target_time - datetime(year, 1, 1)).total_seconds() / 3600 / time_step
                    )
                break
        else:
            return None

    try:
        with h5py.File(h5_path, "r") as f:
            T_total = f["fields"].shape[0]
            if t_idx >= T_total or t_idx < 0:
                return None
            label = f["fields"][t_idx]  # [C_total, H, W]
            label = label[channel_indices]  # [C_selected, H, W]
        return label
    except Exception:
        return None


def compute_metrics(total_files, channel_indices, time_step, data_dir, clim_mean):
    """Compute per-channel RMSE and ACC."""
    n_channels = len(channel_indices)

    if os.path.exists('./result/rmse.npy') and os.path.exists('./result/acc.npy'):
        print("📂 Loading cached metrics...")
        return np.load('./result/rmse.npy'), np.load('./result/acc.npy')

    clim_mean = clim_mean[0, :, :, :]  # (C, H, W)

    channel_rmse = np.zeros(n_channels)
    acc_numerator = np.zeros(n_channels)
    acc_pred_sq = np.zeros(n_channels)
    acc_label_sq = np.zeros(n_channels)
    valid_count = 0

    for file in tqdm(total_files, unit="files", desc="Computing metrics"):
        base_time, lead_hours = parse_pred_filename(file)
        target_time = base_time + timedelta(hours=lead_hours)

        # Load ground truth at target time
        label = get_ground_truth(data_dir, target_time, channel_indices, time_step)
        if label is None:
            continue

        # Load prediction
        pred = np.load(f'result/output/{file}').squeeze()  # (C, H, W)

        if pred.shape != label.shape:
            # Handle shape mismatch
            if pred.ndim == 4:
                pred = pred[0]
            if pred.shape != label.shape:
                continue

        # RMSE
        channel_rmse += np.sqrt(np.mean((label - pred) ** 2, axis=(1, 2)))

        # ACC (Anomaly Correlation Coefficient)
        label_anom = label - clim_mean
        pred_anom = pred - clim_mean
        acc_numerator += np.sum(pred_anom * label_anom, axis=(1, 2))
        acc_pred_sq += np.sum(pred_anom ** 2, axis=(1, 2))
        acc_label_sq += np.sum(label_anom ** 2, axis=(1, 2))

        valid_count += 1

    if valid_count == 0:
        print("⚠️  No valid predictions found.")
        return np.zeros(n_channels), np.zeros(n_channels)

    channel_rmse /= valid_count
    channel_acc = acc_numerator / (np.sqrt(acc_pred_sq * acc_label_sq) + 1e-8)

    np.save('./result/rmse.npy', channel_rmse)
    np.save('./result/acc.npy', channel_acc)
    return channel_rmse, channel_acc


def show_result_table(channels, channel_rmse, channel_acc):
    """Print formatted RMSE/ACC table."""
    w = 40
    print(f"\n┌{'─' * (w + 2)}┬{'─' * 14}┬{'─' * 14}┐")
    print(f"│ {'Channel':<{w}} │ {'RMSE':>12} │ {'ACC':>12} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")

    # Show first 10 + key vars, then average
    display_idx = list(range(min(10, len(channels))))
    # Add key vars if not in first 10
    for key in ['geopotential_500', 'temperature_850', '2m_temperature']:
        if key in channels:
            idx = channels.index(key)
            if idx not in display_idx:
                display_idx.append(idx)

    for i in display_idx:
        ch = channels[i]
        print(f"│ {ch:<{w}} │ {channel_rmse[i]:>12.4f} │ {channel_acc[i]:>12.4f} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    print(f"│ {'Average':<{w}} │ {np.mean(channel_rmse):>12.4f} │ {np.mean(channel_acc):>12.4f} │")
    print(f"└{'─' * (w + 2)}┴{'─' * 14}┴{'─' * 14}┘")


def plot_prediction(label, pred, var, filename):
    """Plot truth, prediction, and difference for a single variable."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
    ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)

    vmin = min(label.min(), pred.min())
    vmax = max(label.max(), pred.max())
    diff = label - pred
    rmse = np.sqrt(np.mean(diff ** 2))
    diff_abs_max = np.abs(diff).max()

    plot_configs = [
        {'data': label, 'title': 'Truth', 'cmap': 'viridis',
         'vmin': vmin, 'vmax': vmax},
        {'data': pred, 'title': 'Prediction', 'cmap': 'viridis',
         'vmin': vmin, 'vmax': vmax},
        {'data': diff, 'title': f'Difference (RMSE={rmse:.2f})',
         'cmap': 'RdBu_r', 'vmin': -diff_abs_max, 'vmax': diff_abs_max},
    ]

    for ax, cfg in zip(axes, plot_configs):
        im = ax.imshow(cfg['data'], cmap=cfg['cmap'],
                       vmin=cfg['vmin'], vmax=cfg['vmax'])
        ax.set_title(cfg['title'], fontsize=12, pad=4)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        plt.colorbar(im, ax=ax, orientation='horizontal')

    fig.suptitle(var, fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_loss, valid_loss):
    """Plot training and validation loss curves."""
    mask = ~(np.isnan(train_loss) | np.isnan(valid_loss))
    train_loss = train_loss[mask]
    valid_loss = valid_loss[mask]

    if len(train_loss) == 0:
        print("⚠️  No loss data to plot.")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {'train': '#2563EB', 'valid': '#EA580C'}
    epochs = np.arange(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, color=colors['train'], linewidth=1.5, label='Train')
    ax.plot(epochs, valid_loss, color=colors['valid'], linewidth=1.5,
            label='Valid', linestyle='--')

    min_idx = np.argmin(valid_loss)
    ax.scatter(epochs[min_idx], valid_loss[min_idx],
               color=colors['valid'], s=40, zorder=5, edgecolors='white')
    ax.annotate(f'Best: {valid_loss[min_idx]:.3f}',
                xy=(epochs[min_idx], valid_loss[min_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, color=colors['valid'],
                arrowprops=dict(arrowstyle='-', color=colors['valid'], lw=0.5))

    ax.set(xlabel='Epoch', ylabel='Loss', xlim=(0, len(train_loss) + 1))
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('./result/loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Loss curves saved to './result/loss.png'")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")

    os.makedirs('./result/', exist_ok=True)

    # ---- Plot loss curves ----
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"
    if os.path.exists(train_loss_file) and os.path.exists(valid_loss_file):
        train_loss = np.load(train_loss_file)
        valid_loss = np.load(valid_loss_file)
        plot_loss_curves(train_loss, valid_loss)
    else:
        print("⚠️  Loss files not found — skipping loss plot.")

    # ---- Compute metrics ----
    data_dir = cfg_data.dataset.data_dir
    total_files, channel_indices, time_step = get_metadata(
        data_dir, cfg_data.dataset.channels
    )

    if len(total_files) == 0:
        print("⚠️  No prediction files in './result/output/' — skipping metrics.")
        sys.exit(0)

    # Load climate mean for ACC
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        mu = f["global_means"][:]
    clim_mean = mu[:, channel_indices, :, :]

    channel_rmse, channel_acc = compute_metrics(
        total_files, channel_indices, time_step, data_dir, clim_mean
    )
    show_result_table(cfg_data.dataset.channels, channel_rmse, channel_acc)

    # ---- Plot example predictions ----
    test_year = cfg_data.dataset.test_time[0]
    eg_files = [f for f in total_files if f.startswith(f'{test_year}')][:3]

    key_vars = ['2m_temperature', 'geopotential_500', 'temperature_500']
    available_vars = [v for v in key_vars if v in cfg_data.dataset.channels]
    channel_index_map = {v: cfg_data.dataset.channels.index(v) for v in available_vars}

    if eg_files:
        print(f"\n📊 Plotting example predictions for: {available_vars}")
        for file in eg_files:
            base_time, lead_hours = parse_pred_filename(file)
            target_time = base_time + timedelta(hours=lead_hours)

            label = get_ground_truth(data_dir, target_time, channel_indices, time_step)
            if label is None:
                print(f"  ⚠️  No ground truth for {file}")
                continue

            pred = np.load(f'result/output/{file}').squeeze()
            if pred.ndim == 4:
                pred = pred[0]

            for var in available_vars:
                idx = channel_index_map[var]
                out_file = f'./result/{file[:-4]}_{var}.png'
                if pred.shape == label.shape:
                    plot_prediction(label[idx], pred[idx], var, out_file)
                    print(f'  ✅ {out_file}')

    print("\n✅ Evaluation complete.")
