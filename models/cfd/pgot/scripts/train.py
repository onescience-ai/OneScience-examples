"""Generic training script for AirfRANS CFD models.

Usage: python train_generic.py --model MODEL_NAME --data-dir ... --output-dir ...

Supported models: khrons, gfocal, pgot, mfscalinglaws
"""

import argparse
import json
import os
import random
import sys
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pyvista as pv


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AirfRANSDataset(Dataset):
    """Direct AirfRANS dataset reader using pyvista."""

    def __init__(self, data_dir, split_name, stats_dir=None):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'manifest.json')) as f:
            manifest = json.load(f)
        self.samples = manifest[split_name]

        self.conditions = {}
        for name in self.samples:
            parts = name.split('_')
            uinf = float(parts[2])
            aoa_deg = float(parts[3])
            aoa_rad = np.deg2rad(aoa_deg)
            vx_inf = uinf * np.cos(aoa_rad)
            vy_inf = uinf * np.sin(aoa_rad)
            self.conditions[name] = (vx_inf, vy_inf)

        self.stats_dir = stats_dir or os.path.join(data_dir, 'stats')
        os.makedirs(self.stats_dir, exist_ok=True)
        self._load_or_compute_stats()

    def _load_or_compute_stats(self):
        paths = {
            'mean_in': os.path.join(self.stats_dir, 'mean_in.npy'),
            'std_in': os.path.join(self.stats_dir, 'std_in.npy'),
            'mean_out': os.path.join(self.stats_dir, 'mean_out.npy'),
            'std_out': os.path.join(self.stats_dir, 'std_out.npy'),
        }
        if all(os.path.exists(p) for p in paths.values()):
            self.mean_in = np.load(paths['mean_in'])
            self.std_in = np.load(paths['std_in'])
            self.mean_out = np.load(paths['mean_out'])
            self.std_out = np.load(paths['std_out'])
            return
        all_x, all_y = [], []
        for name in self.samples[:min(30, len(self.samples))]:
            x, y, _ = self._load_sample(name)
            all_x.append(x)
            all_y.append(y)
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        self.mean_in = all_x.mean(axis=0)
        self.std_in = all_x.std(axis=0) + 1e-8
        self.mean_out = all_y.mean(axis=0)
        self.std_out = all_y.std(axis=0) + 1e-8
        for k, v in {'mean_in': self.mean_in, 'std_in': self.std_in,
                      'mean_out': self.mean_out, 'std_out': self.std_out}.items():
            np.save(paths[k], v)

    def _load_sample(self, name):
        from scipy.spatial import cKDTree
        sample_dir = os.path.join(self.data_dir, name)
        internal = pv.read(os.path.join(sample_dir, f'{name}_internal.vtu'))
        aerofoil = pv.read(os.path.join(sample_dir, f'{name}_aerofoil.vtp'))

        cell_centers = internal.cell_centers()
        pos = np.array(cell_centers.points)[:, :2]
        u = np.array(internal.cell_data['U'])
        p_arr = np.array(internal.cell_data['p'])
        nut = np.array(internal.cell_data['nut'])

        sdf_points = np.array(internal.points)
        sdf_vals = np.array(internal.point_data['implicit_distance'])
        tree = cKDTree(sdf_points[:, :2])
        _, sdf_idx = tree.query(pos)
        sdf = sdf_vals[sdf_idx]

        vx, vy = u[:, 0], u[:, 1]
        vx_inf, vy_inf = self.conditions[name]
        surf_mask = (np.abs(sdf) < 1e-6)

        aero_pos = np.array(aerofoil.points)[:, :2]
        aero_normals = np.array(aerofoil['Normals'])[:, :2]
        n_points = pos.shape[0]
        nx, ny = np.zeros(n_points), np.zeros(n_points)
        surf_indices = np.where(surf_mask)[0]
        if len(surf_indices) > 0 and len(aero_normals) > 0:
            tree2 = cKDTree(aero_pos)
            _, aero_idx = tree2.query(pos[surf_indices])
            nx[surf_indices] = aero_normals[aero_idx, 0]
            ny[surf_indices] = aero_normals[aero_idx, 1]

        x = np.stack([pos[:, 0], pos[:, 1],
                       np.full(n_points, vx_inf), np.full(n_points, vy_inf),
                       sdf, nx, ny], axis=-1)
        y = np.stack([vx, vy, p_arr, nut], axis=-1)
        return x, y, surf_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        x, y, surf = self._load_sample(name)
        x = (x - self.mean_in) / self.std_in
        y = (y - self.mean_out) / self.std_out
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'surf': torch.tensor(surf, dtype=torch.bool),
            'name': name,
        }


MODEL_REGISTRY = {
    'khrons': {'module': 'KHRONOS_model', 'class': 'KHRONOS',
                'init': {'in_dim': 7, 'hidden_dim': 64, 'out_dim': 4}},
    'gfocal': {'module': 'GFocal_model', 'class': 'GFocal',
                'init': {'in_dim': 7, 'hidden_dim': 128, 'out_dim': 4}},
    'pgot': {'module': 'PGOT_model', 'class': 'PGOT',
               'init': {'in_dim': 7, 'hidden_dim': 128, 'out_dim': 4}},
    'mfscalinglaws': {'module': 'MfScalingLaws_model', 'class': 'MfScalingLaws',
                       'init': {'in_dim': 7, 'hidden_dim': 128, 'out_dim': 4}},
}


def build_model(model_name, device):
    info = MODEL_REGISTRY[model_name]
    mod = importlib.import_module(info['module'])
    cls = getattr(mod, info['class'])
    return cls(**info['init']).to(device)


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def train_epoch(model, dataloader, optimizer, device, sample_points):
    model.train()
    total_loss, n = 0.0, 0
    for batch in dataloader:
        x = batch['x'][0].to(device)
        y = batch['y'][0].to(device)
        n_total = x.shape[0]
        if sample_points < n_total:
            idx = torch.randperm(n_total, device=device)[:sample_points]
            x, y = x[idx], y[idx]
        x_in = x.unsqueeze(0)
        y_in = y.unsqueeze(0)
        optimizer.zero_grad()
        pred = model(x_in)
        loss = mse_loss(pred, y_in)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, dataloader, device, sample_points):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in dataloader:
        x = batch['x'][0].to(device)
        y = batch['y'][0].to(device)
        n_total = x.shape[0]
        if sample_points < n_total:
            idx = torch.randperm(n_total, device=device)[:sample_points]
            x, y = x[idx], y[idx]
        x_in = x.unsqueeze(0)
        y_in = y.unsqueeze(0)
        pred = model(x_in)
        total_loss += mse_loss(pred, y_in).item()
        n += 1
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['khrons', 'gfocal', 'pgot', 'mfscalinglaws'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--stats-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--sample-points', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f'[{args.model.upper()}] Device: {device}', flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'[{args.model.upper()}] Loading dataset...', flush=True)
    train_ds = AirfRANSDataset(args.data_dir, 'full_train', args.stats_dir)
    n_train = int(len(train_ds) * 0.9)
    train_sub = torch.utils.data.Subset(train_ds, range(n_train))
    val_sub = torch.utils.data.Subset(train_ds, range(n_train, len(train_ds)))
    train_loader = DataLoader(train_sub, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=1)
    test_ds = AirfRANSDataset(args.data_dir, 'full_test', args.stats_dir)
    test_loader = DataLoader(test_ds, batch_size=1)

    print(f'[{args.model.upper()}] Train:{len(train_sub)} Val:{len(val_sub)} Test:{len(test_ds)}',
          flush=True)

    model = build_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{args.model.upper()}] Params: {n_params:,}', flush=True)

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                  args.sample_points)
        val_loss = validate(model, val_loader, device, args.sample_points)
        print(f'[{args.model.upper()}] E{epoch:3d}/{args.epochs} '
              f'T:{train_loss:.4f} V:{val_loss:.4f}', flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pt'))

    test_loss = validate(model, test_loader, device, args.sample_points)
    print(f'[{args.model.upper()}] Test loss: {test_loss:.4f}', flush=True)

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'final_model.pt'))
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({
            'model': args.model, 'best_val_loss': float(best_val),
            'test_loss': float(test_loss), 'params': n_params,
            'epochs': args.epochs,
        }, f, indent=2)

    print(f'[{args.model.upper()}] Done. Output: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
