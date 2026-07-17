import os
import glob
import numpy as np

from scripts.utils.vtk_io import read_case


class AirfRANSDataPipe:
    """Loads AirfRANS cases for a given split.

    Each item is a dict from utils.vtk_io.read_case.
    """

    def __init__(self, data_root, case_names):
        self.data_root = data_root
        self.case_names = case_names
        self.case_dirs = [os.path.join(data_root, c) for c in case_names]

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        return read_case(self.case_dirs[idx])


def build_normalizer(cases):
    """Compute per-variable z-score statistics over a list of case dicts.

    Variables: d (volume), nx, ny (surface), Vx, Vy (scalar),
               y columns [vx, vy, p, nut] (volume).
    Returns dict of (mean, std) per variable.
    """
    def stats(arrs):
        a = np.concatenate(arrs, axis=0)
        return float(a.mean()), float(a.std() + 1e-8)

    vd = stats([c["d"] for c in cases])
    vnx = stats([c["nx"] for c in cases])
    vny = stats([c["ny"] for c in cases])
    vVx = float(np.mean([c["Vx"] for c in cases])), float(np.std([c["Vx"] for c in cases]) + 1e-8)
    vVy = float(np.mean([c["Vy"] for c in cases])), float(np.std([c["Vy"] for c in cases]) + 1e-8)
    vy = stats([c["y"] for c in cases])
    return {
        "d": vd, "nx": vnx, "ny": vny,
        "Vx": vVx, "Vy": vVy, "y": vy,
    }


def apply_normalizer(case, norm):
    def z(arr, s):
        return ((arr - s[0]) / s[1]).astype(np.float32)
    out = dict(case)
    out["d"] = z(case["d"], norm["d"])
    out["nx"] = z(case["nx"], norm["nx"])
    out["ny"] = z(case["ny"], norm["ny"])
    out["Vx"] = (case["Vx"] - norm["Vx"][0]) / norm["Vx"][1]
    out["Vy"] = (case["Vy"] - norm["Vy"][0]) / norm["Vy"][1]
    out["y"] = z(case["y"], norm["y"])
    return out


def inverse_norm_y(y_norm, norm):
    m, s = norm["y"]
    return y_norm * s + m
