"""Self-contained GeoANF training on AirfRANS using pyvista for VTK reading.

Avoids onescience datapipe dependency chain; reads VTK/VTP files directly.
"""

import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import pyvista as pv

from model import GeoANF


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AirfRANSDataset(Dataset):
    """Direct AirfRANS dataset reader using pyvista.

    Reads VTU internal mesh and VTP aerofoil surface files.
    """

    def __init__(self, data_dir, split_name, stats_dir=None):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'manifest.json')) as f:
            manifest = json.load(f)
        self.samples = manifest[split_name]

        # Parse inflow conditions from sample names
        self.conditions = {}
        for name in self.samples:
            parts = name.split('_')
            # Format: airFoil2D_SST_{Uinf}_{AoA}_{...}
            uinf = float(parts[2])
            aoa_deg = float(parts[3])
            aoa_rad = np.deg2rad(aoa_deg)
            vx_inf = uinf * np.cos(aoa_rad)
            vy_inf = uinf * np.sin(aoa_rad)
            self.conditions[name] = (vx_inf, vy_inf)

        # Load or compute stats
        self.stats_dir = stats_dir or os.path.join(data_dir, 'stats')
        os.makedirs(self.stats_dir, exist_ok=True)
        self._load_or_compute_stats()

    def _load_or_compute_stats(self):
        mean_in_path = os.path.join(self.stats_dir, 'mean_in.npy')
        std_in_path = os.path.join(self.stats_dir, 'std_in.npy')
        mean_out_path = os.path.join(self.stats_dir, 'mean_out.npy')
        std_out_path = os.path.join(self.stats_dir, 'std_out.npy')

        if all(os.path.exists(p) for p in [mean_in_path, std_in_path, mean_out_path, std_out_path]):
            self.mean_in = np.load(mean_in_path)
            self.std_in = np.load(std_in_path)
            self.mean_out = np.load(mean_out_path)
            self.std_out = np.load(std_out_path)
            return

        # Compute stats from training data
        all_x, all_y = [], []
        for name in self.samples[:min(100, len(self.samples))]:  # subsample for speed
            x, y, _ = self._load_sample(name)
            all_x.append(x)
            all_y.append(y)

        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        self.mean_in = all_x.mean(axis=0)
        self.std_in = all_x.std(axis=0) + 1e-8
        self.mean_out = all_y.mean(axis=0)
        self.std_out = all_y.std(axis=0) + 1e-8

        np.save(mean_in_path, self.mean_in)
        np.save(std_in_path, self.std_in)
        np.save(mean_out_path, self.mean_out)
        np.save(std_out_path, self.std_out)

    def _load_sample(self, name):
        sample_dir = os.path.join(self.data_dir, name)
        internal_path = os.path.join(sample_dir, f'{name}_internal.vtu')
        aerofoil_path = os.path.join(sample_dir, f'{name}_aerofoil.vtp')

        internal = pv.read(internal_path)
        aerofoil = pv.read(aerofoil_path)

        # Use cell centers as query points (consistent with cell data)
        cell_centers = internal.cell_centers()
        pos = np.array(cell_centers.points)[:, :2]  # (N_cells, 2)

        # Cell data
        u = np.array(internal.cell_data['U'])       # (N_cells, 3)
        p_arr = np.array(internal.cell_data['p'])    # (N_cells,)
        nut = np.array(internal.cell_data['nut'])    # (N_cells,)

        # Point data for implicit_distance at cell centers
        sdf_points = np.array(internal.points)  # (N_points, 3)
        sdf_vals = np.array(internal.point_data['implicit_distance'])  # (N_points,)

        # Interpolate SDF at cell centers using nearest neighbor
        from scipy.spatial import cKDTree
        tree = cKDTree(sdf_points[:, :2])
        _, sdf_idx = tree.query(pos)
        sdf = sdf_vals[sdf_idx]  # (N_cells,)

        vx, vy = u[:, 0], u[:, 1]

        # Inflow conditions
        vx_inf, vy_inf = self.conditions[name]

        # Surface mask: sdf ≈ 0
        surf_mask = (np.abs(sdf) < 1e-6)

        # Normal vectors from aerofoil VTP
        aero_pos = np.array(aerofoil.points)[:, :2]
        aero_normals = np.array(aerofoil['Normals'])[:, :2]

        n_points = pos.shape[0]
        nx = np.zeros(n_points)
        ny = np.zeros(n_points)

        surf_indices = np.where(surf_mask)[0]
        if len(surf_indices) > 0 and len(aero_normals) > 0:
            tree2 = cKDTree(aero_pos)
            _, aero_idx = tree2.query(pos[surf_indices])
            nx[surf_indices] = aero_normals[aero_idx, 0]
            ny[surf_indices] = aero_normals[aero_idx, 1]

        # Build input features (7-dim)
        x = np.stack([
            pos[:, 0], pos[:, 1],
            np.full(n_points, vx_inf),
            np.full(n_points, vy_inf),
            sdf,
            nx, ny,
        ], axis=-1)  # (N, 7)

        # Build output labels (4-dim)
        y = np.stack([vx, vy, p_arr, nut], axis=-1)  # (N, 4)

        return x, y, surf_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        x, y, surf = self._load_sample(name)

        # Normalize
        x = (x - self.mean_in) / self.std_in
        y = (y - self.mean_out) / self.std_out

        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'surf': torch.tensor(surf, dtype=torch.bool),
            'name': name,
        }


def geoanf_loss(pred, target, surf_mask, alpha=1.0):
    surf = surf_mask
    vol = ~surf
    loss = 0.0
    count = 0
    if surf.any():
        loss += alpha * F.mse_loss(pred[surf], target[surf])
        count += 1
    if vol.any():
        loss += F.mse_loss(pred[vol], target[vol])
        count += 1
    return loss / max(count, 1)


def train_epoch(model, dataloader, optimizer, device, alpha, sample_points):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch['x'][0].to(device)   # (N, 7)
        y = batch['y'][0].to(device)   # (N, 4)
        surf = batch['surf'][0].to(device)  # (N,)

        # Subsample first
        n_total = x.shape[0]
        if sample_points < n_total:
            idx = torch.randperm(n_total, device=device)[:sample_points]
            x = x[idx]
            y = y[idx]
            surf = surf[idx]

        # Extract boundary from subsampled
        boundary = x[surf]              # (m, 7)
        if boundary.shape[0] == 0:
            continue

        boundary = boundary.unsqueeze(0)     # (1, m, 7)
        x_sample = x.unsqueeze(0)             # (1, n_s, 7)
        y_sample = y.unsqueeze(0)             # (1, n_s, 4)

        optimizer.zero_grad()
        pred = model(boundary, x_sample)
        loss = geoanf_loss(pred[0], y_sample[0], surf, alpha)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device, sample_points):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch['x'][0].to(device)
        y = batch['y'][0].to(device)
        surf = batch['surf'][0].to(device)

        # Subsample first
        n_total = x.shape[0]
        if sample_points < n_total:
            idx = torch.randperm(n_total, device=device)[:sample_points]
            x = x[idx]
            y = y[idx]
            surf = surf[idx]

        boundary = x[surf]
        if boundary.shape[0] == 0:
            continue

        boundary = boundary.unsqueeze(0)
        x_sample = x.unsqueeze(0)
        y_sample = y.unsqueeze(0)

        pred = model(boundary, x_sample)
        loss = geoanf_loss(pred[0], y_sample[0], surf, 1.0)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default='./data/airfrans/data/Dataset')
    parser.add_argument('--stats-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str,
                        default='./batch_output/airfrans/GeoANF/output')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--sample-points', type=int, default=16000)
    parser.add_argument('--loss-alpha', type=float, default=1.0)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f'Device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Create datasets
    print('Loading dataset...')
    train_ds = AirfRANSDataset(args.data_dir, 'full_train', args.stats_dir)
    test_ds = AirfRANSDataset(args.data_dir, 'full_test', args.stats_dir)

    # Split train into train/val (90/10)
    n_train = int(len(train_ds) * 0.9)
    train_subset = torch.utils.data.Subset(train_ds, range(n_train))
    val_subset = torch.utils.data.Subset(train_ds, range(n_train, len(train_ds)))

    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f'Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_ds)}')

    # Create model
    model = GeoANF(
        in_dim=7, hidden_dim=args.hidden_dim,
        out_dim=4, num_heads=args.num_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            args.loss_alpha, args.sample_points,
        )
        val_loss, _ = validate(model, val_loader, device, args.sample_points)

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train: {train_loss:.6f} | Val: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pt'))

    # Final test
    print('\nFinal test evaluation...')
    test_loss, _ = validate(model, test_loader, device, args.sample_points)
    print(f'Test loss: {test_loss:.6f}')

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'final_model.pt'))

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({
            'best_val_loss': float(best_val_loss),
            'test_loss': float(test_loss),
            'model_params': n_params,
            'epochs': args.epochs,
        }, f, indent=2)

    print(f'\nDone. Output: {args.output_dir}')


if __name__ == '__main__':
    main()
