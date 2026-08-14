"""Evaluation metrics for AirfRANS flow-field regression.

Implements the paper's LIPS-style metrics:
  - MSE: x-velocity, y-velocity, pressure, turbulent viscosity
  - mean relative drag, mean relative lift
  - Spearman's correlation for drag and lift

Drag/lift are computed per-simulation from surface pressure/velocity following
AirfRANS/LIPS conventions (surface pressure coefficient integration). For a
point-cloud surrogate this uses surface points with known normal vectors.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def mse_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true/pred: (N, 4) = [vx, vy, p, nut]."""
    d = y_pred - y_true
    return {
        "mse_x_velocity": float(np.mean(d[:, 0] ** 2)),
        "mse_y_velocity": float(np.mean(d[:, 1] ** 2)),
        "mse_pressure": float(np.mean(d[:, 2] ** 2)),
        "mse_turbulent_viscosity": float(np.mean(d[:, 3] ** 2)),
    }


def _force_coefficients_per_sim(
    surf_pressure: np.ndarray,
    surf_normals: np.ndarray,
    surf_vel: np.ndarray,
    u_inf: np.ndarray,
    rho: float = 1.0,
) -> tuple[float, float]:
    """Estimate drag and lift coefficients from surface pressure via panel method.

    Args:
        surf_pressure: (N,) pressure on surface points
        surf_normals: (N, 2) outward normals
        surf_vel: (N, 2) velocity on surface points
        u_inf: (2,) inlet velocity vector
        rho: fluid density (arbitrary scale for relative metrics)
    Returns:
        (cd, cl) drag and lift coefficients
    """
    # free-stream speed
    u_mag = float(np.linalg.norm(u_inf))
    if u_mag < 1e-8:
        return 0.0, 0.0
    # dynamic pressure q = 0.5 * rho * u^2
    q = 0.5 * rho * u_mag * u_mag
    if q == 0:
        return 0.0, 0.0
    # force = -p * n  (pressure acts inward => force outward = -p*n on surface)
    # integrate over surface points (approximate with mean*count, unit length)
    n = len(surf_pressure)
    if n == 0:
        return 0.0, 0.0
    fx = -np.mean(surf_pressure * surf_normals[:, 0])
    fy = -np.mean(surf_pressure * surf_normals[:, 1])
    # unit direction of free-stream
    udir = u_inf / u_mag
    # drag along stream, lift perpendicular (rotate -90deg: (uy, -ux)? use +90)
    # stream dir e_x = udir; lift dir e_y = (-udir[1], udir[0])
    drag = fx * udir[0] + fy * udir[1]
    lift = -fx * udir[1] + fy * udir[0]
    cd = drag / q
    cl = lift / q
    return cd, cl


def _surface_sampling_per_sim(
    data_dir: str,
    sim_name: str,
    pred_scaled: np.ndarray,
    scaler_out_mean: np.ndarray,
    scaler_out_scale: np.ndarray,
    u_inf: np.ndarray,
):
    """Return (pressure_true, pressure_pred, normals, vel_true, vel_pred) for a sim.

    Loads surface points from the aerofoil vtp and maps predictions via
    nearest-neighbour on the internal grid.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy.spatial import cKDTree

    import os
    sim_dir = os.path.join(data_dir, sim_name)

    # internal
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(os.path.join(sim_dir, f"{sim_name}_internal.vtu"))
    r.Update()
    g = r.GetOutput()
    pd = g.GetPointData()
    pos = vtk_to_numpy(g.GetPoints().GetData())[:, :2]
    n_internal = g.GetNumberOfPoints()
    p_true = vtk_to_numpy(pd.GetArray("p"))
    U_true = vtk_to_numpy(pd.GetArray("U"))[:, :2]

    # surface
    rs = vtk.vtkXMLPolyDataReader()
    rs.SetFileName(os.path.join(sim_dir, f"{sim_name}_aerofoil.vtp"))
    rs.Update()
    gs = rs.GetOutput()
    pds = gs.GetPointData()
    sp = vtk_to_numpy(gs.GetPoints().GetData())[:, :2]
    norms = vtk_to_numpy(pds.GetArray("Normals"))[:, :2]

    tree = cKDTree(pos)
    _, idx = tree.query(sp)

    # surface predictions are the appended tail of the dataset point list
    surf_pred = pred_scaled[n_internal:]

    p_true_surf = p_true[idx]
    p_pred_surf = surf_pred[:, 2] * scaler_out_scale[2] + scaler_out_mean[2]
    vel_true_surf = U_true[idx]
    vel_pred_surf = surf_pred[:, :2] * scaler_out_scale[:2] + scaler_out_mean[:2]
    return p_true_surf, p_pred_surf, norms, vel_true_surf, vel_pred_surf

def physics_metrics(
    data_dir: str,
    sim_names: list[str],
    preds_scaled: list[np.ndarray],
    scaler_out_mean: np.ndarray,
    scaler_out_scale: np.ndarray,
    inlet_velocities: list[np.ndarray],
) -> dict:
    """Compute per-sim drag/lift and their relative/Spearman metrics.

    Args:
        preds_scaled: per-sim predictions in standardized space (N_sim, 4)
        inlet_velocities: per-sim u_inf (2,)
    """
    cd_true, cl_true, cd_pred, cl_pred = [], [], [], []
    for i, sim in enumerate(sim_names):
        # true and predicted surface pressure/velocity
        p_t, p_p, norms, v_t, v_p = _surface_sampling_per_sim(
            data_dir, sim, preds_scaled[i], scaler_out_mean, scaler_out_scale,
            inlet_velocities[i],
        )
        u_inf = inlet_velocities[i]
        cd_t, cl_t = _force_coefficients_per_sim(p_t, norms, v_t, u_inf)
        cd_p, cl_p = _force_coefficients_per_sim(p_p, norms, v_p, u_inf)
        cd_true.append(cd_t)
        cl_true.append(cl_t)
        cd_pred.append(cd_p)
        cl_pred.append(cl_p)

    cd_true = np.array(cd_true)
    cl_true = np.array(cl_true)
    cd_pred = np.array(cd_pred)
    cl_pred = np.array(cl_pred)

    def rel_mean(gt, pr):
        denom = np.abs(gt) + 1e-9
        return float(np.mean(np.abs(pr - gt) / denom))

    def spearman(a, b):
        if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
            return float("nan")
        return float(spearmanr(a, b).statistic)

    return {
        "mean_relative_drag": rel_mean(cd_true, cd_pred),
        "mean_relative_lift": rel_mean(cl_true, cl_pred),
        "spearman_drag": spearman(cd_true, cd_pred),
        "spearman_lift": spearman(cl_true, cl_pred),
    }


def summarize(
    data_dir: str,
    sim_names: list[str],
    preds_scaled: list[np.ndarray],
    true_scaled: list[np.ndarray],
    scaler_out_mean: np.ndarray,
    scaler_out_scale: np.ndarray,
    inlet_velocities: list[np.ndarray],
) -> dict:
    """Aggregate MSE + physics metrics for a dataset split."""
    mse_all = []
    true_denorm_all, pred_denorm_all = [], []
    for p, t in zip(preds_scaled, true_scaled):
        p_d = p * scaler_out_scale + scaler_out_mean
        t_d = t * scaler_out_scale + scaler_out_mean
        mse_all.append(mse_metrics(t_d, p_d))
        true_denorm_all.append(t_d)
        pred_denorm_all.append(p_d)
    mse_agg = {k: float(np.mean([m[k] for m in mse_all])) for k in mse_all[0]}

    phys = physics_metrics(
        data_dir, sim_names, preds_scaled, scaler_out_mean, scaler_out_scale,
        inlet_velocities,
    )
    result = {"mse": mse_agg}
    result.update(phys)
    return result
