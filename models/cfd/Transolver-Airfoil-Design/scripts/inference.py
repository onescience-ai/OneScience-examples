from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams

from train import build_datapipe, build_model, resolve_path


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("transolver-inference")


def load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run scripts/train.py first or provide a compatible weight.")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_fake_inference(cfg_model, cfg_data, cfg_train, logger: logging.Logger, device: torch.device) -> int:
    model_params = cfg_model.specific_params[cfg_model.name]
    datapipe = build_datapipe(cfg_data, model_params, distributed=False)
    test_loader = datapipe._loader(datapipe.test_dataset, False)[0]

    model = build_model(cfg_model).to(device)
    checkpoint_path = resolve_path(cfg_train.checkpoint_dir) / f"{cfg_model.name}.pth"
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    result_dir = resolve_path(cfg_train.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    mean_out, std_out = datapipe.coef_norm[2], datapipe.coef_norm[3]
    preds, targets = [], []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = out.cpu().numpy() * (std_out + 1e-8) + mean_out
            target = data.y.cpu().numpy() * (std_out + 1e-8) + mean_out
            preds.append(pred)
            targets.append(target)

    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    mse = float(criterion(torch.from_numpy(pred_arr), torch.from_numpy(target_arr)))
    np.savez(result_dir / "predictions.npz", pred=pred_arr, target=target_arr, mse=np.array(mse, dtype=np.float32))
    (result_dir / "score.json").write_text(json.dumps({"model_name": cfg_model.name, "mse": mse}, indent=2), encoding="utf-8")
    logger.info("Saved fake inference outputs to %s", result_dir)
    logger.info("MSE: %.6f", mse)
    return 0


def run_airfrans_inference(cfg_model, cfg_data, cfg_train, logger: logging.Logger, device: torch.device) -> int:
    import pyvista as pv
    import scipy.stats as sc
    from tqdm import tqdm
    from onescience.utils.transolver.metrics import (
        Airfoil_test,
        Compute_coefficients,
        Infer_test,
        NumpyEncoder,
        rel_err,
    )

    model_params = cfg_model.specific_params[cfg_model.name]
    hparams = model_params
    hparams["subsampling"] = cfg_data.data.subsampling
    datapipe = build_datapipe(cfg_data, model_params, distributed=False)
    coef_norm = datapipe.coef_norm
    test_loader = datapipe.test_dataloader()
    test_names = datapipe.test_dataset.data_list_names

    model = build_model(cfg_model).to(device)
    checkpoint_path = resolve_path(cfg_train.checkpoint_dir) / f"{cfg_model.name}.pth"
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    path_in = resolve_path(cfg_data.source.data_dir)
    path_out = resolve_path(cfg_train.result_dir) / cfg_data.data.splits.task
    path_out.mkdir(parents=True, exist_ok=True)
    sample_count = min(int(cfg_train.n_test), len(test_names))
    chosen = sorted(random.sample(range(len(test_names)), k=sample_count)) if sample_count else []

    criterion = nn.MSELoss(reduction="none")
    scores_vol, scores_surf, scores_force, scores_p, scores_wss = [], [], [], [], []
    times, true_coefs, pred_coefs = [], [], []

    for j, data in enumerate(tqdm(test_loader, desc="Testing")):
        sim_name = test_names[j]
        u_inf, angle = float(sim_name.split("_")[2]), float(sim_name.split("_")[3])
        outs, elapsed = Infer_test(device, [model], [hparams], data, coef_norm=coef_norm)
        times.append(elapsed)

        intern = pv.read(path_in / sim_name / f"{sim_name}_internal.vtu")
        aerofoil = pv.read(path_in / sim_name / f"{sim_name}_aerofoil.vtp")
        tc, true_intern, true_airfoil = Compute_coefficients([intern], [aerofoil], data.surf.cpu(), u_inf, angle, keep_vtk=True)
        tc, true_airfoil = tc[0], true_airfoil[0]
        intern_pred, aerofoil_pred = Airfoil_test(intern, aerofoil, outs, coef_norm, data.surf.cpu())
        pc, intern_pred_vtk, aerofoil_pred_vtk = Compute_coefficients(intern_pred, aerofoil_pred, data.surf.cpu(), u_inf, angle, keep_vtk=True)
        true_coefs.append(tc)
        pred_coefs.append(pc)

        out = outs[0]
        scores_vol.append(criterion(out[~data.surf], data.y[~data.surf]).mean(dim=0).cpu().numpy())
        scores_surf.append(criterion(out[data.surf], data.y[data.surf]).mean(dim=0).cpu().numpy())
        scores_force.append(rel_err(tc, pc[0]))
        scores_wss.append(rel_err(true_airfoil.point_data["wallShearStress"], aerofoil_pred_vtk[0].point_data["wallShearStress"]).mean(axis=0))
        scores_p.append(rel_err(true_airfoil.point_data["p"], aerofoil_pred_vtk[0].point_data["p"]).mean(axis=0))
        if j in chosen:
            intern_pred_vtk[0].save(path_out / f"{sim_name}_pred_internal.vtu")
            aerofoil_pred_vtk[0].save(path_out / f"{sim_name}_pred_aerofoil.vtp")

    true_coefs_arr = np.array(true_coefs)
    pred_coefs_arr = np.array(pred_coefs)
    spear = [
        sc.spearmanr(true_coefs_arr[:, 0], pred_coefs_arr[:, 0, 0])[0],
        sc.spearmanr(true_coefs_arr[:, 1], pred_coefs_arr[:, 0, 1])[0],
    ]
    score = {
        "model_name": cfg_model.name,
        "mean_time": np.array(times).mean(axis=0),
        "std_time": np.array(times).std(axis=0),
        "mean_score_vol": np.array(scores_vol).mean(axis=0),
        "mean_score_surf": np.array(scores_surf).mean(axis=0),
        "mean_score_force": np.array(scores_force).mean(axis=0),
        "mean_rel_p": np.array(scores_p).mean(axis=0),
        "mean_rel_wss": np.array(scores_wss).mean(axis=0),
        "spearman_coef": np.array(spear),
    }
    (path_out / f"score_{cfg_model.name}.json").write_text(json.dumps(score, indent=2, cls=NumpyEncoder), encoding="utf-8")
    logger.info("Saved AirfRANS inference outputs to %s", path_out)
    return 0


def main() -> int:
    config_path = ROOT / "conf" / "config.yaml"
    cfg_model = YParams(str(config_path), "model")
    cfg_data = YParams(str(config_path), "datapipe")
    cfg_train = YParams(str(config_path), "training")
    logger = setup_logging()
    DistributedManager.initialize()
    device_name = getattr(cfg_train, "device", "auto")
    if device_name == "cpu":
        device = torch.device("cpu")
    elif device_name.startswith("cuda"):
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{cfg_train.gpuid}" if torch.cuda.is_available() else "cpu")
    if cfg_data.backend == "fake_airfrans":
        return run_fake_inference(cfg_model, cfg_data, cfg_train, logger, device)
    if cfg_data.backend == "airfrans":
        return run_airfrans_inference(cfg_model, cfg_data, cfg_train, logger, device)
    raise ValueError(f"Unsupported datapipe.backend: {cfg_data.backend}")


if __name__ == "__main__":
    raise SystemExit(main())
