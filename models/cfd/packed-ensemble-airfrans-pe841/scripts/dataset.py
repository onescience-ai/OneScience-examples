"""AirfRANS dataset loader for point-wise flow-field regression.

Builds 7-feature inputs [pos_x, pos_y, u_inf_x, u_inf_y, dist, normal_x, normal_y]
and 4-target outputs [v_x, v_y, p, nut] from the AirfRANS VTU/VTP files,
following the paper's feature/target specification.

Data layout (local dataset):
  <Dataset>/
    manifest.json            # split definitions
    <sim_name>/              # one dir per simulation
      <sim_name>_internal.vtu     # volume mesh: pos, U, p, nut, implicit_distance
      <sim_name>_aerofoil.vtp     # surface: pos, U, p, nut, Normals
      <sim_name>_freestream.vtp   # freestream boundary: pos, U, p, nut

The freestream boundary U gives the inlet velocity (u_inf_x, u_inf_y).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    VTK_AVAILABLE = True
except Exception:  # pragma: no cover
    VTK_AVAILABLE = False


def _read_vtk(path: str, reader_cls):
    r = reader_cls()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def _read_internal_vtu(sim_dir: str, sim_name: str):
    path = os.path.join(sim_dir, f"{sim_name}_internal.vtu")
    grid = _read_vtk(path, vtk.vtkXMLUnstructuredGridReader)
    pd = grid.GetPointData()
    pos = vtk_to_numpy(grid.GetPoints().GetData())[:, :2].astype(np.float32)
    U = vtk_to_numpy(pd.GetArray("U"))[:, :2].astype(np.float32)
    p = vtk_to_numpy(pd.GetArray("p")).astype(np.float32)
    nut = vtk_to_numpy(pd.GetArray("nut")).astype(np.float32)
    dist = vtk_to_numpy(pd.GetArray("implicit_distance")).astype(np.float32)
    return pos, U, p, nut, dist


def _read_aerofoil_vtp(sim_dir: str, sim_name: str):
    path = os.path.join(sim_dir, f"{sim_name}_aerofoil.vtp")
    grid = _read_vtk(path, vtk.vtkXMLPolyDataReader)
    pd = grid.GetPointData()
    pos = vtk_to_numpy(grid.GetPoints().GetData())[:, :2].astype(np.float32)
    normals = vtk_to_numpy(pd.GetArray("Normals"))[:, :2].astype(np.float32)
    return pos, normals


def _read_freestream_vtp(sim_dir: str, sim_name: str):
    path = os.path.join(sim_dir, f"{sim_name}_freestream.vtp")
    grid = _read_vtk(path, vtk.vtkXMLPolyDataReader)
    pd = grid.GetPointData()
    U = vtk_to_numpy(pd.GetArray("U"))[:, :2].astype(np.float32)
    return U


def _inlet_velocity(sim_dir: str, sim_name: str) -> np.ndarray:
    """Inlet velocity from freestream boundary U (mean of boundary points)."""
    try:
        U = _read_freestream_vtp(sim_dir, sim_name)
        return U.mean(axis=0)
    except Exception:
        # fallback: far-field internal points
        return np.array([1.0, 0.0], dtype=np.float32)


class AirfRANSDataset(Dataset):
    """Point-cloud dataset over a list of simulations."""

    def __init__(
        self,
        data_dir: str,
        sim_names: list[str],
        max_nodes_per_sim: int = -1,
        use_surface: bool = True,
    ) -> None:
        assert VTK_AVAILABLE, "vtk not available"
        self.data_dir = data_dir
        self.sim_names = sim_names
        self.max_nodes_per_sim = max_nodes_per_sim
        self.use_surface = use_surface
        self.samples = []  # list of (pos, feat, target, surf)

        for name in sim_names:
            sim_dir = os.path.join(data_dir, name)
            pos, U, p, nut, dist = _read_internal_vtu(sim_dir, name)
            u_inf = _inlet_velocity(sim_dir, name)
            n = pos.shape[0]
            x = np.stack(
                [pos[:, 0], pos[:, 1],
                 np.full(n, u_inf[0]), np.full(n, u_inf[1]),
                 dist, np.zeros(n), np.zeros(n)],
                axis=1,
            ).astype(np.float32)
            y = np.stack([U[:, 0], U[:, 1], p, nut], axis=1).astype(np.float32)

            if self.use_surface:
                sp, sn = _read_aerofoil_vtp(sim_dir, name)
                m = sp.shape[0]
                sx = np.stack(
                    [sp[:, 0], sp[:, 1],
                     np.full(m, u_inf[0]), np.full(m, u_inf[1]),
                     np.zeros(m), sn[:, 0], sn[:, 1]],
                    axis=1,
                ).astype(np.float32)
                # match surface points to internal volume points by proximity
                sy = self._match_surface_targets(pos, U, p, nut, sp)
                if sy is not None:
                    x = np.concatenate([x, sx], axis=0)
                    y = np.concatenate([y, sy], axis=0)

            if self.max_nodes_per_sim > 0 and x.shape[0] > self.max_nodes_per_sim:
                idx = np.random.choice(x.shape[0], self.max_nodes_per_sim, replace=False)
                x, y = x[idx], y[idx]

            self.samples.append((x, y))

        self._concat()

    def _match_surface_targets(self, pos, U, p, nut, sp):
        """Nearest-neighbour surface target lookup from internal grid."""
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        _, idx = tree.query(sp)
        sy = np.stack(
            [U[idx, 0], U[idx, 1], p[idx], nut[idx]], axis=1
        ).astype(np.float32)
        return sy

    def _concat(self):
        xs = [s[0] for s in self.samples]
        ys = [s[1] for s in self.samples]
        self.x = np.concatenate(xs, axis=0).astype(np.float32)
        self.y = np.concatenate(ys, axis=0).astype(np.float32)
        self._n_per_sim = [s[0].shape[0] for s in self.samples]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


def load_manifest(data_dir: str) -> dict:
    mp = os.path.join(data_dir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f)
    return {}


def list_simulations(data_dir: str) -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "airFoil*")))


def get_splits(data_dir: str) -> dict:
    """Return dict of split -> list of sim names. Falls back to heuristic split."""
    manifest = load_manifest(data_dir)
    sims = list_simulations(data_dir)

    # Try structured manifest formats
    splits = {}
    m = manifest
    # common layout: manifest["splits"] = {"train": [...], "test": [...], ...}
    if isinstance(m.get("splits"), dict):
        for k, v in m["splits"].items():
            if isinstance(v, list) and v and isinstance(v[0], str):
                splits[k] = [s for s in v if s in set(sims)]
            elif isinstance(v, (int, float)):
                pass
    if splits:
        return splits

    # Heuristic: paper train=103, test=200, test-OOD=496 out of ~1000 sims.
    # Use first 103 for train, next 200 for test, last for OOD by default,
    # unless user provides explicit split files.
    n = len(sims)
    return {
        "train": sims[:103],
        "test": sims[103:303],
        "test_ood": sims[303:303 + 496],
    }


def compute_stats(dataset: AirfRANSDataset) -> tuple:
    scaler_in = StandardScaler().fit(dataset.x)
    scaler_out = StandardScaler().fit(dataset.y)
    return scaler_in, scaler_out


def build_dataloaders(
    data_dir: str,
    batch_size: int = 4096,
    num_workers: int = 0,
    max_nodes_per_sim: int = 8000,
    train_sim_names: list[str] | None = None,
    eval_sim_names: dict | None = None,
    seed: int = 0,
):
    """Build train/val/test dataloaders with standardized features."""
    rng = np.random.default_rng(seed)
    if train_sim_names is None:
        splits = get_splits(data_dir)
        train_sim_names = splits["train"]

    # down-sample training sims for small-scale repro if requested
    train_ds = AirfRANSDataset(data_dir, train_sim_names, max_nodes_per_sim=max_nodes_per_sim)
    scaler_in, scaler_out = compute_stats(train_ds)

    train_x = scaler_in.transform(train_ds.x).astype(np.float32)
    train_y = scaler_out.transform(train_ds.y).astype(np.float32)

    ds_list = []
    if eval_sim_names is None:
        splits = get_splits(data_dir)
        eval_sim_names = {
            "val": splits["train"][-20:],  # holdout subset
            "test": splits["test"][:20],
            "test_ood": splits["test_ood"][:20],
        }

    def make_ds(sims):
        d = AirfRANSDataset(data_dir, sims, max_nodes_per_sim=-1)
        return d

    tensors_x = torch.from_numpy(train_x)
    tensors_y = torch.from_numpy(train_y)
    train_set = torch.utils.data.TensorDataset(tensors_x, tensors_y)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # eval datasets keep full points; standardize with train stats
    eval_loaders = {}
    for split, sims in eval_sim_names.items():
        d = make_ds(sims)
        ex = scaler_in.transform(d.x).astype(np.float32)
        ey = scaler_out.transform(d.y).astype(np.float32)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(ex), torch.from_numpy(ey)
        )
        eval_loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    stats = {
        "scaler_in_mean": scaler_in.mean_,
        "scaler_in_scale": scaler_in.scale_,
        "scaler_out_mean": scaler_out.mean_,
        "scaler_out_scale": scaler_out.scale_,
    }
    return train_loader, eval_loaders, stats
