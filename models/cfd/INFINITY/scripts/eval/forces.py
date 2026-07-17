import numpy as np
from scipy.spatial import cKDTree


def compute_forces(surf_x, surf_normals, vol_x, pred_y_norm, Vx=1.0, Vy=0.0, rho_ref=1.225):
    """Compute drag/lift coefficients from predicted fields at surface nodes.

    ASSUMPTION: standard aerodynamic coefficients using surface pressure and a
    velocity-based viscous proxy. Coordinates are 2D; surface normals provided.
    Returns (CD, CL) coefficients (ASSUMPTION-based integration, not the exact
    AirfRANS integral but a physically-motivated proxy for trend comparison).

    Args:
      surf_x: (M,2) surface node coords
      surf_normals: (M,2) surface unit normals
      vol_x: (N,2) volume node coords
      pred_y_norm: (N,4) predicted [vx,vy,p,nut] at volume nodes (normalized)
      Vx,Vy: inlet velocity (used for dynamic pressure scale)
    """
    p = pred_y_norm[:, 2]
    vx = pred_y_norm[:, 0]
    vy = pred_y_norm[:, 1]

    tree = cKDTree(vol_x)
    _, idx = tree.query(surf_x)
    p_s = p[idx]
    vx_s = vx[idx]
    vy_s = vy[idx]

    # dynamic pressure from inlet speed (physical proxy)
    V_inf = float(np.sqrt(Vx ** 2 + Vy ** 2) + 1e-8)
    q = 0.5 * rho_ref * V_inf ** 2 + 1e-8

    # pressure force per unit span: F = -p * n
    fpress = -p_s[:, None] * surf_normals  # (M,2)
    # viscous proxy: tangential momentum along freestream direction
    t_dir = np.array([Vx, Vy], dtype=np.float64)
    t_dir = t_dir / (np.linalg.norm(t_dir) + 1e-8)
    vt = vx_s * t_dir[0] + vy_s * t_dir[1]
    fvisc = (vt[:, None] * t_dir[None, :]) * 1e-3
    f = fpress + fvisc
    fx = float(np.sum(f[:, 0]))
    fy = float(np.sum(f[:, 1]))
    CD = fx / (q + 1e-8)
    CL = fy / (q + 1e-8)
    return CD, CL
