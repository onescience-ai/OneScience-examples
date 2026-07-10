from __future__ import annotations

import logging
import os
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from model import Transolver3D, Transolver3D_plus
import onescience
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.utils.transolver import cal_coefficient, save_prediction_to_vtk, visualize_prediction


def load_shapenet_car_datapipe():
    module_path = Path(onescience.__file__).resolve().parent / "datapipes/cfd/ShapeNetCar.py"
    spec = importlib.util.spec_from_file_location("_onescience_shapenetcar", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ShapeNetCarDatapipe from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ShapeNetCarDatapipe


def setup_logging(rank: int) -> logging.Logger:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger(__name__)


def build_model(model_name: str, model_params, device: torch.device) -> torch.nn.Module:
    model_cls = {
        "Transolver": Transolver3D,
        "Transolver_plus": Transolver3D_plus,
    }.get(model_name)
    if model_cls is None:
        raise NotImplementedError(f"Model {model_name} initialization not implemented.")
    return model_cls(
        n_hidden=model_params.n_hidden,
        n_layers=model_params.n_layers,
        space_dim=model_params.space_dim,
        fun_dim=model_params.fun_dim,
        n_head=model_params.n_head,
        mlp_ratio=model_params.mlp_ratio,
        out_dim=model_params.out_dim,
        slice_num=model_params.slice_num,
        unified_pos=model_params.unified_pos,
    ).to(device)


def resolve_device(gpuid: int) -> torch.device:
    if torch.cuda.is_available() and int(gpuid) >= 0:
        return torch.device(f"cuda:{gpuid}")
    return torch.device("cpu")


def maybe_calculate_coefficient(data_dir: Path, pred_press: np.ndarray, pred_velo: np.ndarray, gt_press: np.ndarray, gt_velo: np.ndarray):
    if not (data_dir / "quadpress_smpl.vtk").exists() or not (data_dir / "hexvelo_smpl.vtk").exists():
        return None, None
    pred_coef = cal_coefficient(str(data_dir), pred_press[:, None], pred_velo)
    gt_coef = cal_coefficient(str(data_dir), gt_press[:, None], gt_velo)
    return pred_coef, gt_coef


def main() -> None:
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)
    if manager.rank != 0:
        logger.warning("Inference should run on a single process; exiting non-zero rank.")
        return

    config_file_path = str(ROOT / "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")
    cfg_test = YParams(config_file_path, "inference")

    model_name = cfg.name
    model_params = cfg.specific_params[model_name]
    cfg_data.model_hparams = model_params

    device = resolve_device(cfg_test.gpuid)
    logger.info("Using device: %s", device)

    ShapeNetCarDatapipe = load_shapenet_car_datapipe()
    datapipe = ShapeNetCarDatapipe(params=cfg_data, distributed=False)
    val_dataset = datapipe.val_dataset
    coef_norm = datapipe.coef_norm
    val_names = val_dataset.data_list_names
    test_loader, _ = datapipe.val_dataloader()
    logger.info("Loaded %d validation samples.", len(val_dataset))

    model = build_model(model_name, model_params, device)
    checkpoint_path = Path(cfg_train.checkpoint_dir) / f"{model_name}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    result_root = Path(cfg_test.result_dir) / model_name
    npy_dir = result_root / "npy"
    vtk_dir = result_root / "vtk"
    vis_dir = result_root / "vis"
    npy_dir.mkdir(parents=True, exist_ok=True)
    if cfg_test.save_vtk:
        vtk_dir.mkdir(parents=True, exist_ok=True)
    if cfg_test.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    criterion_func = nn.MSELoss(reduction="none")
    l2errs_press, l2errs_velo, mses_press, mses_velo_var, times = [], [], [], [], []
    gt_coef_list, pred_coef_list = [], []

    mean = torch.tensor(coef_norm[2], dtype=torch.float32, device=device)
    std = torch.tensor(coef_norm[3], dtype=torch.float32, device=device)

    with torch.no_grad():
        for index, data in enumerate(test_loader):
            if index >= len(val_names):
                break
            sample_name = val_names[index]
            data = data.to(device)
            tic = time.time()
            out = model(data)
            times.append(time.time() - tic)
            targets = data.y

            pred_press = out[data.surf, -1] * std[-1] + mean[-1]
            gt_press = targets[data.surf, -1] * std[-1] + mean[-1]
            pred_velo = out[~data.surf, :-1] * std[:-1] + mean[:-1]
            gt_velo = targets[~data.surf, :-1] * std[:-1] + mean[:-1]
            out_denorm = out * std + mean
            y_denorm = targets * std + mean

            safe_name = sample_name.replace("/", "_")
            np.save(npy_dir / f"{index}_{safe_name}_pred.npy", out_denorm.cpu().numpy())
            np.save(npy_dir / f"{index}_{safe_name}_gt.npy", y_denorm.cpu().numpy())

            data_dir = ROOT / cfg_data.source.data_dir / sample_name
            pred_coef, gt_coef = maybe_calculate_coefficient(
                data_dir,
                pred_press.cpu().numpy(),
                pred_velo.cpu().numpy(),
                gt_press.cpu().numpy(),
                gt_velo.cpu().numpy(),
            )
            if pred_coef is not None and gt_coef is not None:
                pred_coef_list.append(pred_coef)
                gt_coef_list.append(gt_coef)

            l2errs_press.append((torch.norm(pred_press - gt_press) / (torch.norm(gt_press) + 1e-8)).cpu().numpy())
            l2errs_velo.append((torch.norm(pred_velo - gt_velo) / (torch.norm(gt_velo) + 1e-8)).cpu().numpy())
            mses_press.append(criterion_func(out[data.surf, -1], targets[data.surf, -1]).mean().cpu().numpy())
            mses_velo_var.append(criterion_func(out[~data.surf, :-1], targets[~data.surf, :-1]).mean().cpu().numpy())

            if cfg_test.save_vtk and (data_dir / "quadpress_smpl.vtk").exists():
                save_prediction_to_vtk(
                    out_denorm=out_denorm,
                    targets=targets,
                    cfd_data=data,
                    sample_name=sample_name,
                    output_dir=str(vtk_dir),
                    index=index,
                    data_dir=str(ROOT / cfg_data.source.data_dir),
                )
            if cfg_test.visualize and cfg_test.save_vtk:
                visualize_prediction(output_dir=str(vtk_dir), vis_dir=str(vis_dir), index=index)

    logger.info("Results saved to: %s", result_root)
    logger.info("Relative L2 pressure: %.6f", float(np.mean(l2errs_press)))
    logger.info("Relative L2 velocity: %.6f", float(np.mean(l2errs_velo)))
    logger.info("RMSE pressure: %.6f", float(np.sqrt(np.mean(mses_press)) * coef_norm[3][-1]))
    rmse_velo = np.sqrt(np.mean(mses_velo_var, axis=0)) * coef_norm[3][:-1]
    logger.info("Combined velocity RMSE: %.6f", float(np.sqrt(np.mean(np.square(rmse_velo)))))
    logger.info("Mean inference time (s): %.6f", float(np.mean(times)))
    if gt_coef_list:
        coef_error = np.mean(np.abs(np.array(pred_coef_list) - np.array(gt_coef_list)) / (np.array(gt_coef_list) + 1e-8))
        logger.info("Mean relative CD error: %.6f", float(coef_error))
    else:
        logger.info("Skipped drag coefficient metrics because VTK geometry files were not present.")


if __name__ == "__main__":
    main()
