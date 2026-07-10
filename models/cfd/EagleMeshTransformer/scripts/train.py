import ctypes
import logging
import os
import random
import sys
import sysconfig
import time
from pathlib import Path


def preload_python_shared_library():
    """Make libpython visible to native extensions loaded with ctypes."""
    libdir = sysconfig.get_config_var("LIBDIR")
    version = sysconfig.get_config_var("VERSION")
    if not libdir or not version:
        return

    candidates = [
        Path(libdir) / f"libpython{version}.so.1.0",
        Path(libdir) / f"libpython{version}.so",
    ]
    for libpython in candidates:
        if libpython.exists():
            ctypes.CDLL(str(libpython), mode=ctypes.RTLD_GLOBAL)
            return


preload_python_shared_library()

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

# 获取项目根目录（train.py上级的上级）
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from model.graphViT import GraphViT
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.datapipes.cfd import EagleDatapipe



def save_best_model(model, optimizer, checkpoint_dir: str):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        Path(checkpoint_dir) / "best_model.pth",
    )


def load_best_model(model, checkpoint_dir: str, device: torch.device):
    ckpt_path = Path(checkpoint_dir) / "best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def setup_logging(rank: int):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("train")


def get_loss(velocity, pressure, output, state_hat, target, mask, alpha):
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]
    velocity_hat = state_hat[:, 1:, :, :2]
    pressure_hat = state_hat[:, 1:, :, 2:]
    mask = mask[:, 1:].unsqueeze(-1)

    loss_velocity = torch.sqrt(((velocity * mask - velocity_hat * mask) ** 2).mean(dim=-1)).mean()
    loss_pressure = torch.sqrt(((pressure * mask - pressure_hat * mask) ** 2).mean(dim=-1)).mean()
    mse = nn.MSELoss()
    loss = mse(target[..., :2] * mask, output[..., :2] * mask)
    loss = loss + alpha * mse(target[..., 2:] * mask, output[..., 2:] * mask)
    return {"loss": loss, "MSE_velocity": loss_velocity, "MSE_pressure": loss_pressure}


def move_batch(x, device):
    return {
        "mesh_pos": x["mesh_pos"].to(device),
        "edges": x["edges"].to(device).long(),
        "velocity": x["velocity"].to(device),
        "pressure": x["pressure"].to(device),
        "node_type": x["node_type"].to(device),
        "mask": x["mask"].to(device),
        "cluster": x["cluster"].to(device).long(),
        "cluster_mask": x["cluster_mask"].to(device).long(),
    }


def fix_single_cluster_path(datapipe, cfg_data):
    if int(cfg_data.data.n_cluster) != 1:
        return

    cluster_path = Path(cfg_data.source.cluster_dir)
    for dataset_name in ("train_dataset", "val_dataset", "test_dataset"):
        dataset = getattr(datapipe, dataset_name, None)
        if dataset is not None and getattr(dataset, "cluster_path", None) is None:
            dataset.cluster_path = cluster_path


def validate(model, dataloader, device, alpha, manager):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x in dataloader:
            if not x:
                continue
            batch = move_batch(x, device)
            state = torch.cat([batch["velocity"], batch["pressure"]], dim=-1)
            state_hat, output, target = model(
                batch["mesh_pos"],
                batch["edges"],
                state,
                batch["node_type"],
                batch["cluster"],
                batch["cluster_mask"],
                apply_noise=False,
            )
            dataset = dataloader.dataset
            state_hat[..., :2], state_hat[..., 2:] = dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            velocity, pressure = dataset.denormalize(batch["velocity"], batch["pressure"])
            costs = get_loss(velocity, pressure, output, state_hat, target, batch["mask"], alpha)
            if manager.world_size > 1:
                dist.all_reduce(costs["loss"], op=dist.ReduceOp.AVG)
            total_loss += costs["loss"].item()
            count += 1
    return total_loss / max(count, 1)


def main():
    os.chdir(root_path)
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)

    config_path = root_path / "config" / "config.yaml"
    cfg_model = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")
    cfg_train = YParams(config_path, "training")

    seed = int(cfg_train.get("seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    datapipe = EagleDatapipe(params=cfg_data, distributed=(manager.world_size > 1))
    fix_single_cluster_path(datapipe, cfg_data)
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()

    device_name = cfg_train.get("device", "auto")
    device = manager.device if device_name == "auto" else torch.device(device_name)
    model = GraphViT(state_size=cfg_model.state_size, w_size=cfg_model.w_size).to(device)
    if manager.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[manager.local_rank], output_device=manager.local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_train.lr))
    best_valid_loss = float("inf")
    best_epoch = 0

    for epoch in range(int(cfg_train.max_epoch)):
        start = time.time()
        if manager.world_size > 1:
            train_sampler.set_epoch(epoch)
            if val_sampler:
                val_sampler.set_epoch(epoch)

        model.train()
        train_loss, count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=(manager.rank != 0))
        for x in pbar:
            if not x:
                continue
            batch = move_batch(x, device)
            state = torch.cat([batch["velocity"], batch["pressure"]], dim=-1)
            state_hat, output, target = model(
                batch["mesh_pos"],
                batch["edges"],
                state,
                batch["node_type"],
                batch["cluster"],
                batch["cluster_mask"],
                apply_noise=True,
            )
            state_hat[..., :2], state_hat[..., 2:] = train_loader.dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            velocity, pressure = train_loader.dataset.denormalize(batch["velocity"], batch["pressure"])
            costs = get_loss(velocity, pressure, output, state_hat, target, batch["mask"], cfg_train.loss_alpha)

            optimizer.zero_grad()
            costs["loss"].backward()
            optimizer.step()

            train_loss += costs["loss"].item()
            count += 1
            pbar.set_postfix(loss=f"{costs['loss'].item():.6f}")

        train_loss /= max(count, 1)
        valid_loss = validate(model, val_loader, device, cfg_train.loss_alpha, manager)
        if manager.rank == 0:
            logger.info(
                "Epoch %s/%s | %.2fs | train %.6f | valid %.6f",
                epoch + 1,
                cfg_train.max_epoch,
                time.time() - start,
                train_loss,
                valid_loss,
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                save_best_model(model, optimizer, cfg_train.checkpoint_dir)
                logger.info("Saved checkpoint to %s/best_model.pth", cfg_train.checkpoint_dir)
            if epoch - best_epoch > int(cfg_train.patience):
                break

    if manager.rank == 0:
        final_model = GraphViT(state_size=cfg_model.state_size, w_size=cfg_model.w_size).to(device)
        load_best_model(final_model, cfg_train.checkpoint_dir, device)
        test_loader, _ = datapipe.test_dataloader()
        test_loss = validate(final_model, test_loader, device, cfg_train.loss_alpha, manager)
        logger.info("Final test loss: %.6f", test_loss)

    manager.cleanup()


if __name__ == "__main__":
    main()
