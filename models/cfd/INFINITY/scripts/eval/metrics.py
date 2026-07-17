import numpy as np
from scipy.stats import spearmanr
from scripts.eval.inference import infer_case
from scripts.data.airfrans_dataset import apply_normalizer


def evaluate(model, pipe, norm, cfg, n_pts=None, n_cases=None):
    if n_pts is None:
        n_pts = cfg["sampling"]["n_pts"]

    vol_mse = {u: [] for u in ["vx", "vy", "p", "nut"]}

    idxs = range(len(pipe)) if n_cases is None else range(min(n_cases, len(pipe)))

    for i in idxs:
        raw = pipe[i]
        case = apply_normalizer(raw, norm)
        pred_norm, true_norm, vol_idx = infer_case(model, case, cfg, n_pts)
        for j, u in enumerate(["vx", "vy", "p", "nut"]):
            vol_mse[u].append(float(np.mean((pred_norm[:, j] - true_norm[:, j]) ** 2)))

    out = {"volume_mse": {u: float(np.mean(v)) for u, v in vol_mse.items()},
           "n_cases": len(vol_mse["vx"])}
    return out
