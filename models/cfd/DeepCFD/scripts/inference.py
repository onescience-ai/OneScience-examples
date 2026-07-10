import os
import sys
import importlib.util
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.utils.deepcfd.functions import visualize
import onescience


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config():
    cfg = YParams(str(PROJECT_ROOT / "config" / "config.yaml"), "root")
    cfg.datapipe.source.data_dir = str(resolve_path(cfg.datapipe.source.data_dir))
    cfg.inference.checkpoint_path = str(resolve_path(cfg.inference.checkpoint_path))
    cfg.inference.result_dir = str(resolve_path(cfg.inference.result_dir))
    return cfg


def load_deepcfd_datapipe_class():
    runtime_root = Path(onescience.__file__).resolve().parent
    datapipe_file = runtime_root / "datapipes" / "cfd" / "deepcfd.py"
    spec = importlib.util.spec_from_file_location("_onescience_deepcfd_datapipe", datapipe_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DeepCFD datapipe from {datapipe_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DeepCFDDatapipe


def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    cfg = load_config()
    DeepCFDDatapipe = load_deepcfd_datapipe_class()

    checkpoint_path = Path(cfg.inference.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", cfg.model.to_dict())
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    datapipe = DeepCFDDatapipe(cfg.datapipe, distributed=False)
    test_loader, _ = datapipe.test_dataloader()
    batch = next(iter(test_loader))

    x = batch["x"].to(device)
    y = batch["y"].to(device)
    with torch.no_grad():
        out = model(x)

    error = torch.abs(out.cpu() - y.cpu())
    mse = torch.mean((out.cpu() - y.cpu()) ** 2, dim=(0, 2, 3)).numpy()
    mae = torch.mean(error, dim=(0, 2, 3)).numpy()

    result_dir = Path(cfg.inference.result_dir)
    vis_dir = result_dir / "vis_results"
    pred_dir = result_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    np.save(pred_dir / "prediction_batch.npy", out.cpu().numpy())
    np.save(pred_dir / "target_batch.npy", y.cpu().numpy())
    np.save(pred_dir / "absolute_error_batch.npy", error.numpy())

    y_np = y.cpu().numpy()
    out_np = out.cpu().numpy()
    err_np = error.numpy()
    for i in range(min(cfg.inference.num_visualize, x.shape[0])):
        visualize(y_np, out_np, err_np, i, save_dir=str(vis_dir))

    if dist.rank == 0:
        print(f"Checkpoint: {checkpoint_path}")
        print(f"MSE per channel [Ux, Uy, p]: {mse}")
        print(f"MAE per channel [Ux, Uy, p]: {mae}")
        print(f"Results saved to {result_dir}")

    dist.cleanup()


if __name__ == "__main__":
    main()
