import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
import glob
import h5py
from datetime import datetime
from tqdm import tqdm
from onescience.utils.fcn.YParams import YParams
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 0.9
rcParams['xtick.major.width'] = 0.9
rcParams['ytick.major.width'] = 0.9


def get_metadata(data_dir, channels):
    """Read variable list and time_step from h5 attrs."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [
            v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]
        ]
        time_step = int(ds.attrs["time_step"])

    channel_indices = [all_variables.index(v) for v in channels]

    total_files = [f for f in os.listdir('./result/output/') if f.endswith('.npy')]
    total_files.sort()
    return total_files, channel_indices, time_step


def filename_to_index(filename, time_step):
    """Convert YYYYMMDDHH filename to time step index within the year's h5."""
    dt = datetime.strptime(filename, "%Y%m%d%H")
    year_start = datetime(dt.year, 1, 1)
    hours = (dt - year_start).total_seconds() / 3600
    return int(hours / time_step)


def get_result(total_files, channel_indices, time_step, data_dir, clim_mean):
    """Compute per-channel RMSE and ACC (latitude-weighted, matching official ClimaX).

    Official lat_weighted_acc in climax/utils/metrics.py:
    1. De-normalizes pred and y
    2. Subtracts climatology: pred_anom = pred - clim, y_anom = y - clim
    3. Centers by per-field spatial mean: pred_prime = pred_anom - mean(pred_anom)
    4. Applies latitude weighting (cos(lat) / mean(cos(lat)))
    5. Computes: sum(w * pred_prime * y_prime) / sqrt(sum(w * pred_prime^2) * sum(w * y_prime^2))

    Here we compute an unweighted version since lat array is not readily available;
    for RMSE we use the standard (non-lat-weighted) formula for simplicity.
    """
    channel_rmse = np.zeros(len(channel_indices))
    channel_acc = np.zeros(len(channel_indices))
    clim_mean = clim_mean[0, :, :, :]

    if not os.path.exists('./result/rmse.npy') or not os.path.exists('result/acc.npy'):
        numerator = np.zeros(len(channel_indices))
        pred_sq_sum = np.zeros(len(channel_indices))
        label_sq_sum = np.zeros(len(channel_indices))

        for file in tqdm(total_files, unit="files"):
            fname = file[:-4]  # remove .npy
            year = fname[:4]
            t_idx = filename_to_index(fname, time_step)
            with h5py.File(os.path.join(data_dir, 'data', f'{year}.h5'), "r") as f:
                label = f["fields"][t_idx]  # [C, H, W]
                label = label[channel_indices]
            pred = np.load(f'result/output/{file}').squeeze()

            # RMSE computation
            channel_rmse += np.sqrt(np.mean((label - pred) ** 2, axis=(1, 2)))

            # ACC computation (following official ClimaX lat_weighted_acc without lat weighting)
            label_anom = label - clim_mean
            pred_anom = pred - clim_mean
            # Center by per-field spatial mean (official ClimaX approach)
            pred_prime = pred_anom - np.mean(pred_anom, axis=(1, 2), keepdims=True)
            label_prime = label_anom - np.mean(label_anom, axis=(1, 2), keepdims=True)
            # accumulate
            numerator += np.sum(pred_prime * label_prime, axis=(1, 2))
            pred_sq_sum += np.sum(pred_prime ** 2, axis=(1, 2))
            label_sq_sum += np.sum(label_prime ** 2, axis=(1, 2))

        channel_rmse /= len(total_files)
        channel_acc = numerator / (np.sqrt(pred_sq_sum * label_sq_sum) + 1e-8)
        np.save('./result/acc.npy', channel_acc)
        np.save('./result/rmse.npy', channel_rmse)


def show_result():
    """Print formatted RMSE/ACC table."""
    channel_rmse = np.load('./result/rmse.npy')
    channel_acc = np.load('./result/acc.npy')

    channels = [cfg_data.dataset.out_variables[i]
                for i in range(len(channel_indices))]
    w = 30

    print(f"\n{'=' * (w + 32)}")
    print(f"ClimaX Evaluation Results")
    print(f"{'=' * (w + 32)}")
    print(f"┌{'─' * (w + 2)}┬{'─' * 14}┬{'─' * 14}┐")
    print(f"│ {'Channel':<{w}} │ {'RMSE':>12} │ {'ACC':>12} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    for i, ch in enumerate(channels):
        print(f"│ {ch:<{w}} │ {channel_rmse[i]:>12.4f} │ {channel_acc[i]:>12.4f} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    print(f"│ {'Average':<{w}} │ {np.mean(channel_rmse):>12.4f} │ {np.mean(channel_acc):>12.4f} │")
    print(f"└{'─' * (w + 2)}┴{'─' * 14}┴{'─' * 14}┘")


def plot(label, pred, var, filename):
    """Plot truth, prediction, and difference for a single variable."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
    ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)

    # unified color scale for truth/pred
    vmin = min(label.min(), pred.min())
    vmax = max(label.max(), pred.max())

    diff = label - pred
    rmse = np.sqrt(np.mean(diff ** 2))
    diff_abs_max = np.abs(diff).max()

    plot_configs = [
        {'data': label, 'title': 'Truth', 'cmap': 'viridis',
         'vmin': vmin, 'vmax': vmax},
        {'data': pred,  'title': 'Prediction', 'cmap': 'viridis',
         'vmin': vmin, 'vmax': vmax},
        {'data': diff,  'title': f'Difference (RMSE={rmse:.2f})',
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


def plot_loss(train_loss, valid_loss):
    """Plot training and validation loss curves."""
    mask = ~(np.isnan(train_loss) | np.isnan(valid_loss))
    train_loss = train_loss[mask]
    valid_loss = valid_loss[mask]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {'train': '#2563EB', 'valid': '#EA580C'}
    epochs = np.arange(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, color=colors['train'],
            linewidth=1.5, label='Train')
    ax.plot(epochs, valid_loss, color=colors['valid'],
            linewidth=1.5, label='Valid', linestyle='--')

    min_idx = np.argmin(valid_loss)
    ax.scatter(epochs[min_idx], valid_loss[min_idx],
               color=colors['valid'], s=40, zorder=5, edgecolors='white')
    ax.annotate(
        f'Best: {valid_loss[min_idx]:.3f}',
        xy=(epochs[min_idx], valid_loss[min_idx]),
        xytext=(10, 10), textcoords='offset points',
        fontsize=8, color=colors['valid'],
        arrowprops=dict(arrowstyle='-', color=colors['valid'], lw=0.5)
    )

    ax.set(xlabel='Epoch', ylabel='Loss', xlim=(0, len(train_loss) + 1))
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('./result/loss.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")

    train_loss = np.load(f'{cfg.checkpoint_dir}/trloss.npy')
    valid_loss = np.load(f'{cfg.checkpoint_dir}/valoss.npy')
    plot_loss(train_loss, valid_loss)

    data_dir = cfg_data.dataset.data_dir
    out_vars = cfg_data.dataset.out_variables
    total_files, channel_indices, time_step = get_metadata(data_dir, out_vars)

    # Load & compute RMSE/ACC per channel
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        mu = f["global_means"][:]
    clim_mean = mu[:, channel_indices, :, :]
    get_result(total_files, channel_indices, time_step, data_dir, clim_mean)
    show_result()

    ##### Plot example predictions #####
    test_year = cfg_data.dataset.test_time[0]
    eg_files = [f'{test_year}010206']
    selected_vars = out_vars[:3]  # first 3 output variables

    print(f"\nPlotting example predictions for: {eg_files}")
    print(f"Variables: {selected_vars}")

    for file in eg_files:
        year = file[:4]
        t_idx = filename_to_index(file, time_step)
        with h5py.File(os.path.join(data_dir, 'data', f'{year}.h5'), "r") as f:
            label = f["fields"][t_idx]
            label = label[channel_indices]
        pred = np.load(f'result/output/{file}.npy').squeeze()

        for i, var in enumerate(selected_vars):
            filename = f'./result/{file}_{var}.png'
            plot(label[i], pred[i], var, filename)
            print(f'  Saved: {filename}')
