import os

import pyvista as pv
import numpy as np


def read_case(case_dir):
    """Read a single AirfRANS case and return geometry inputs + targets.

    Returns dict with numpy arrays:
      x        : (N, 2) volume node coords (xy)
      d        : (N, 1) implicit distance (volume)
      nx, ny   : (M, 1) surface normals (x,y)
      surf_x   : (M, 2) surface node coords
      Vx, Vy   : scalar inlet velocity (mean freestream U)
      y        : (N, 4) target fields [vx, vy, p, nut]
    """
    name = os.path.basename(case_dir.rstrip("/"))
    internal = pv.read(os.path.join(case_dir, f"{name}_internal.vtu"))
    aerofoil = pv.read(os.path.join(case_dir, f"{name}_aerofoil.vtp"))
    freestream = pv.read(os.path.join(case_dir, f"{name}_freestream.vtp"))

    x = internal.points[:, :2].astype(np.float64)            # (N,2)
    d = internal.point_data["implicit_distance"].astype(np.float64).reshape(-1, 1)
    U = internal.point_data["U"].astype(np.float32)
    p = internal.point_data["p"].astype(np.float32).reshape(-1, 1)
    nut = internal.point_data["nut"].astype(np.float32).reshape(-1, 1)
    y = np.concatenate([U[:, :2], p, nut], axis=1).astype(np.float32)  # (N,4)

    normals = aerofoil.point_data["Normals"].astype(np.float32)
    surf_x = aerofoil.points[:, :2].astype(np.float64)
    nx = normals[:, 0:1].astype(np.float32)
    ny = normals[:, 1:2].astype(np.float32)

    free_U = freestream.point_data["U"].astype(np.float32)
    Vx = float(np.mean(free_U[:, 0]))
    Vy = float(np.mean(free_U[:, 1]))

    return {
        "x": x, "d": d, "nx": nx, "ny": ny, "surf_x": surf_x,
        "Vx": Vx, "Vy": Vy, "y": y,
    }
