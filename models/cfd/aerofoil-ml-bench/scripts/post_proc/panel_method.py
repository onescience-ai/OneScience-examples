import math

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay

from normalise import denormalise_ys


def integrate(vertices, om_values):
    tri = Delaunay(vertices)
    triangles = vertices[tri.simplices]
    om_triangles = om_values[tri.simplices]
    total = 0.0
    for triangle, om in zip(triangles, om_triangles):
        v1, v2, v3 = triangle
        area = 0.5 * abs(
            (v1[0] * (v2[1] - v3[1]) + v2[0] * (v3[1] - v1[1]) + v3[0] * (v1[1] - v2[1]))
        )
        total += area * np.mean(om)
    return total


def lift_coef(val_outs, coef_norm):
    """Calculate lift coefficient via panel method from predicted vorticity.

    NOTE: the demonstration dataset maps the 5 Q channels to
    [rho, rho_u, rho_v, e, omega]. omega is channel index 4.
    """
    rmse = 0
    mse = 0
    out_list = []
    it = 0
    # val_outs is a list of batches; each batch is a list of data_list objects (or a Data)
    for d in val_outs:
        if isinstance(d, (list, tuple)):
            data = d[0]
        else:
            data = d
        cl_target = data.cl if hasattr(data, 'cl') else None
        preds = data.x
        omega = preds[:, 4]
        x = data.pos[:, 0]
        surf = data.surf

        x_foil = np.array(x[surf].cpu())
        del_list = [i for i in range(len(x_foil)) if x_foil[i] < 0]
        pos = np.array(data.pos[surf].cpu())
        omega_foil = np.array(omega[surf].cpu())
        pos = np.delete(pos, del_list, axis=0)
        omega_foil = np.delete(omega_foil, del_list, axis=0)

        if pos.shape[0] < 3:
            # cannot triangulate; record NaN for this sample
            out_list.append([getattr(data, 'foil_n', it), getattr(data, 'alpha', 0),
                             float('nan'), cl_target, None])
            it += 1
            continue

        Gam = integrate(pos, omega_foil)
        a = 340.15
        U = 0.1 * a
        cl = torch.tensor([(2 * Gam) / U])

        if cl_target is not None:
            loss = nn.MSELoss()
            mse_loss = loss(cl, cl_target)
            root = math.sqrt(mse_loss)
            mse += mse_loss.cpu().numpy()
            rmse += root
        else:
            root = float('nan')

        foil_n = getattr(data, 'foil_n', it)
        alpha = getattr(data, 'alpha', 0)
        out_list.append([foil_n, alpha, root, cl_target, cl])
        it += 1

    rmse_avg = rmse / it if it else float('nan')
    mse_avg = mse / it if it else float('nan')
    return rmse_avg, mse_avg, out_list
