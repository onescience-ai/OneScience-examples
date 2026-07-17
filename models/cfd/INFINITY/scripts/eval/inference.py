import os
import json
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utils.config import load_config
from scripts.data.airfrans_dataset import AirfRANSDataPipe, apply_normalizer
from model.infinity import INFINITY


def _sample(coords, values, n_pts):
    N = coords.shape[0]
    if N > n_pts:
        idx = np.random.choice(N, n_pts, replace=False)
        return coords[idx], values[idx], idx
    return coords, values, np.arange(N)


def infer_case(model, case, cfg, n_pts):
    """Encode -> Process -> Decode on a sampled subset of nodes.

    Returns (pred_norm (S,4), true_norm (S,4), idx) on the sampled points.
    """
    device = next(model.parameters()).device
    K = cfg["inr_train"]["K_inner"]
    inner_lr = cfg["inr_train"]["inner_lr"]

    cx, cd, _ = _sample(case["x"], case["d"][:, 0], n_pts)
    sn = np.concatenate([case["nx"], case["ny"]], 1)
    cs, csn, _ = _sample(case["surf_x"], sn, n_pts)
    x = torch.tensor(cx, dtype=torch.float32, device=device)
    surf_x = torch.tensor(cs, dtype=torch.float32, device=device)
    n_in = torch.tensor(csn, dtype=torch.float32, device=device)

    zd = model.encode(model.inrs["d"], x, torch.tensor(cd, dtype=torch.float32, device=device), inner_lr, K)
    zn = model.encode(model.inrs["n"], surf_x, n_in, inner_lr, K)
    z_out = model.process(zd, zn, case["Vx"], case["Vy"])
    pred = model.predict_fields(x, z_out).detach().cpu().numpy()  # (S,4) normalized

    cy, cyv, idx = _sample(case["x"], case["y"], n_pts)
    return pred, cyv, idx


def load_model(cfg, device="cpu"):
    ckpt_full = torch.load(os.path.join(cfg["defaults"]["out_dir"], "infinity_full.pt"),
                           map_location=device, weights_only=False)
    inr_ckpt = torch.load(os.path.join(cfg["defaults"]["out_dir"], "infinity_inr.pt"),
                          map_location=device, weights_only=False)
    norm = ckpt_full["norm"]
    model = INFINITY(cfg)
    model.inrs.load_state_dict(inr_ckpt["inr_state"])
    model.g_psi.load_state_dict(ckpt_full["g_psi_state"])
    model.to(device)
    model.eval()
    return model, norm


if __name__ == "__main__":
    import sys
    from scripts.eval.metrics import evaluate

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/infinity.yaml"
    cfg = load_config(cfg_path)
    manifest = cfg["defaults"]["manifest"]
    with open(manifest) as f:
        m = json.load(f)
    test_names = m[cfg["defaults"]["test_split"]]

    model, norm = load_model(cfg, "cpu")
    pipe = AirfRANSDataPipe(cfg["defaults"]["data_root"], test_names)
    res = evaluate(model, pipe, norm, cfg, n_pts=cfg["sampling"]["n_pts"])
    print(json.dumps(res, indent=2))
