"""Dataset loader for the demonstration reproduction.

The paper's official dataset (NREL windAI_bench airfoil_2k) is an 834GB H5 file
not reachable at usable speed from this environment.  Per decision (方案C) we use
the locally-available CFD_Benchmark NACA cylinder airfoil data
(CFD_Benchmark/airfoil: NACA_Cylinder_{X,Y,Q}.npy) as a stand-in to demonstrate
the paper's models and pipeline end-to-end.

Each grid sample (structured 221x51 grid) is converted to a torch_geometric Data
object with the paper's input/output channel semantics:
    input  x (6 ch): [pos_x, pos_y, rho_inf, sig_u_inf, sig_v_inf, -dist]
    output y (5 ch): [rho, rho_u, rho_v, e, omega]
    pos, surf (surface marker), edge_index (radius graph for graph models).

NOTE: this is a *demonstration* dataset, not the paper's original windAI_bench.
Results are not directly comparable to Tables 1-4.
"""

import os

import numpy as np
import torch
import torch_geometric.nn as nng
from torch_geometric.data import Data


def _radial_distance(pos_2d, center=(0.5, 0.0)):
    return np.sqrt((pos_2d[:, 0] - center[0]) ** 2 + (pos_2d[:, 1] - center[1]) ** 2)


def build_dataset(source_dir, out_channels=5, in_channels=6, r=0.05, max_neighbors=6,
                  graph=True, subsample=None, seed=42, n_samples=None):
    """Load NACA_Cylinder data and build a list of PyG Data objects.

    Args:
        source_dir: directory containing NACA_Cylinder_{X,Y,Q}.npy
        graph: if True, build radius_graph edge_index (needed for GraphSAGE/GUNet)
    Returns:
        list[Data]
    """
    X = np.load(os.path.join(source_dir, 'NACA_Cylinder_X.npy'))  # (N, 221, 51)
    Y = np.load(os.path.join(source_dir, 'NACA_Cylinder_Y.npy'))
    Q = np.load(os.path.join(source_dir, 'NACA_Cylinder_Q.npy'))  # (N, 5, 221, 51)

    N = X.shape[0]
    if n_samples is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, size=min(n_samples, N), replace=False)
        X, Y, Q = X[idx], Y[idx], Q[idx]
        N = X.shape[0]

    # Flatten grid to nodes per sample
    h, w = X.shape[1], X.shape[2]
    rng = np.random.RandomState(seed)
    rho_inf, Ma_inf = 1.0, 0.1
    # free-stream momentum components (reference direction = x-axis)
    sig_u_inf = Ma_inf
    sig_v_inf = 0.0

    dataset = []
    for i in range(N):
        pos = np.stack([X[i].reshape(-1), Y[i].reshape(-1)], axis=1).astype(np.float32)  # (HW, 2)
        n = pos.shape[0]

        # pick subset if requested
        if subsample is not None and n > subsample:
            sel = rng.choice(n, size=subsample, replace=False)
            pos = pos[sel]

        dist = _radial_distance(pos)
        neg_dist = -dist.astype(np.float32)

        # outputs: map the 5 Q channels to [rho, rho_u, rho_v, e, omega]
        if subsample is not None and n > subsample:
            Qf = Q[i].reshape(5, -1).T.astype(np.float32)[sel]
        else:
            Qf = Q[i].reshape(5, -1).T.astype(np.float32)
        y = Qf  # (n, 5)

        # inputs: [pos_x, pos_y, rho_inf, sig_u_inf, sig_v_inf, -dist]
        x = np.zeros((pos.shape[0], in_channels), dtype=np.float32)
        x[:, 0] = pos[:, 0]
        x[:, 1] = pos[:, 1]
        x[:, 2] = rho_inf
        x[:, 3] = sig_u_inf
        x[:, 4] = sig_v_inf
        x[:, 5] = neg_dist

        pos_t = torch.tensor(pos)
        x_t = torch.tensor(x)
        y_t = torch.tensor(y)

        # surface marker: points close to cylinder (small radial distance)
        surf = torch.tensor((_radial_distance(pos) < 0.02).astype(np.bool_))

        if graph:
            edge_index = nng.radius_graph(x=pos_t, r=r, loop=True, max_num_neighbors=max_neighbors)
            data = Data(pos=pos_t, x=x_t, y=y_t, surf=surf, edge_index=edge_index)
        else:
            data = Data(pos=pos_t, x=x_t, y=y_t, surf=surf)
        dataset.append(data)
    return dataset


def make_dataset(source_dir, out_channels=5, in_channels=6, r=0.05, max_neighbors=6,
                 graph=True, subsample=10000, seed=42, n_samples=None):
    return build_dataset(
        source_dir=source_dir, out_channels=out_channels, in_channels=in_channels,
        r=r, max_neighbors=max_neighbors, graph=graph,
        subsample=subsample, seed=seed, n_samples=n_samples,
    )
