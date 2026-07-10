from __future__ import annotations

import logging
import os
import sys
import time
import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from model import Transolver3D, Transolver3D_plus
import onescience
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams


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


def save_checkpoint(model, optimizer, scheduler, epoch: int, loss: float, ckp_dir: str, model_name: str) -> None:
    Path(ckp_dir).mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        Path(ckp_dir) / f"{model_name}.pth",
    )


def main() -> None:
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)

    config_file_path = str(ROOT / "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")

    model_name = cfg.name
    if model_name not in cfg.specific_params:
        raise ValueError(f"Model '{model_name}' not found in model.specific_params.")
    model_params = cfg.specific_params[model_name]
    cfg_data.model_hparams = model_params

    logger.info("Initializing ShapeNetCar datapipe...")
    ShapeNetCarDatapipe = load_shapenet_car_datapipe()
    datapipe = ShapeNetCarDatapipe(params=cfg_data, distributed=(manager.world_size > 1))
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()

    if manager.world_size > 1:
        device = torch.device(f"cuda:{manager.local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = resolve_device(cfg_train.gpuid)

    model = build_model(model_name, model_params, device)
    if manager.rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model: %s, trainable params: %.2fM", model_name, total_params / 1e6)

    if manager.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[manager.local_rank],
            output_device=manager.local_rank,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg_train.lr,
        total_steps=max(1, len(train_dataloader) * cfg_train.max_epoch),
    )
    if cfg_train.loss_criterion == "MSE":
        loss_criterion = nn.MSELoss(reduction="none")
    elif cfg_train.loss_criterion == "MAE":
        loss_criterion = nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Unknown loss criterion: {cfg_train.loss_criterion}")

    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    logger.info("Starting training...")

    for epoch in range(cfg_train.max_epoch):
        epoch_start_time = time.time()
        if manager.world_size > 1:
            train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

        model.train()
        train_loss = train_loss_press = train_loss_velo = 0.0
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            targets = data.y
            loss_press = loss_criterion(out[data.surf, -1], targets[data.surf, -1]).mean()
            loss_velo = loss_criterion(out[:, :-1], targets[:, :-1]).mean()
            loss = loss_velo + cfg_train.loss_weight * loss_press
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_loss_press += loss_press.item()
            train_loss_velo += loss_velo.item()

        train_loss /= max(1, len(train_dataloader))
        train_loss_press /= max(1, len(train_dataloader))
        train_loss_velo /= max(1, len(train_dataloader))

        valid_loss = valid_loss_press = valid_loss_velo = 0.0
        if (epoch + 1) % cfg_train.val_iter == 0 or epoch == cfg_train.max_epoch - 1:
            model.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    data = data.to(device)
                    out = model(data)
                    targets = data.y
                    loss_press = loss_criterion(out[data.surf, -1], targets[data.surf, -1]).mean()
                    loss_velo = loss_criterion(out[:, :-1], targets[:, :-1]).mean()
                    loss = loss_velo + cfg_train.loss_weight * loss_press
                    if manager.world_size > 1:
                        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                        dist.all_reduce(loss_press, op=dist.ReduceOp.AVG)
                        dist.all_reduce(loss_velo, op=dist.ReduceOp.AVG)
                    valid_loss += loss.item()
                    valid_loss_press += loss_press.item()
                    valid_loss_velo += loss_velo.item()

            valid_loss /= max(1, len(val_dataloader))
            valid_loss_press /= max(1, len(val_dataloader))
            valid_loss_velo /= max(1, len(val_dataloader))

        if manager.rank == 0:
            logger.info(
                "Epoch [%d/%d] | Time: %.2fs | Train: %.6f (velo %.6f, press %.6f) | "
                "Valid: %.6f (velo %.6f, press %.6f)",
                epoch + 1,
                cfg_train.max_epoch,
                time.time() - epoch_start_time,
                train_loss,
                train_loss_velo,
                train_loss_press,
                valid_loss,
                valid_loss_velo,
                valid_loss_press,
            )
            if valid_loss > 0 and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_loss_epoch = epoch
                save_checkpoint(model, optimizer, scheduler, epoch, valid_loss, cfg_train.checkpoint_dir, model_name)
                logger.info("New best checkpoint saved to %s/%s.pth", cfg_train.checkpoint_dir, model_name)
            if epoch - best_loss_epoch > cfg_train.patience:
                logger.warning("Validation loss has not improved for %d epochs. Stopping.", cfg_train.patience)
                break


if __name__ == "__main__":
    main()
