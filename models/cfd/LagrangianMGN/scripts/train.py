import logging
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from common import load_config
from onescience.datapipes.cfd import DeepMindLagrangianDatapipe
from onescience.distributed.manager import DistributedManager
from onescience.launch.utils import load_checkpoint, save_checkpoint


def setup_logging(rank: int):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("lagrangian_mgn.train")


def resolve_device(requested: str, manager: DistributedManager):
    if manager.distributed:
        return manager.device
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_step(model, graph, dataset, criterion, optimizer, scaler, scheduler, device, amp_enabled):
    graph = graph.to(device)
    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type=device.type, enabled=amp_enabled):
        gt_pos, gt_vel, gt_acc = dataset.unpack_targets(graph)
        pred_acc = model(graph.ndata["x"], graph.edata["x"], graph)

        mask = graph.ndata["mask"].unsqueeze(-1).to(pred_acc.dtype)
        num_nz = torch.clamp(mask.sum() * dataset.dim, min=1.0)
        loss_acc_norm = (mask * criterion(pred_acc, gt_acc)).sum() / num_nz

        with torch.no_grad():
            pos, vel, _ = dataset.unpack_inputs(graph)
            pred_pos, pred_vel = dataset.time_integrator(
                position=pos,
                velocity=vel[-1],
                acceleration=pred_acc,
                dt=dataset.dt,
                denormalize=True,
            )
            loss_pos = (mask * criterion(pred_pos, gt_pos)).sum() / num_nz
            loss_vel = (
                mask * criterion(pred_vel, dataset.denormalize_velocity(gt_vel))
            ).sum() / num_nz
            loss_acc = (
                mask
                * criterion(
                    dataset.denormalize_acceleration(pred_acc),
                    dataset.denormalize_acceleration(gt_acc),
                )
            ).sum() / num_nz

    scaler.scale(loss_acc_norm).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    return {
        "loss": loss_acc_norm.item() + loss_pos.item() + loss_vel.item(),
        "loss_acc_norm": loss_acc_norm.item(),
        "loss_pos": loss_pos.item(),
        "loss_vel": loss_vel.item(),
        "loss_acc": loss_acc.item(),
    }


def main():
    cfg = load_config()
    Path(cfg.output).mkdir(parents=True, exist_ok=True)
    Path(cfg.resume_dir).mkdir(parents=True, exist_ok=True)

    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)

    if manager.rank == 0:
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    datapipe = DeepMindLagrangianDatapipe(cfg, distributed=manager.distributed)
    train_loader = datapipe.train_dataloader()
    val_loader = datapipe.val_dataloader()
    dataset = datapipe.train_dataset

    if cfg.model.recompute_activation and str(cfg.model.mlp_activation_fn).lower() != "silu":
        raise ValueError("recompute_activation only supports SiLU.")

    device = resolve_device(str(cfg.train.device), manager)
    model = hydra.utils.instantiate(cfg.model).to(device)
    if cfg.compile.enabled:
        model = torch.compile(model, **cfg.compile.args)
    if manager.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[manager.local_rank],
            output_device=manager.local_rank,
            find_unused_parameters=False,
        )

    criterion = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    num_iterations = max(int(cfg.train.epochs) * len(train_loader), 1)
    if cfg.lr_scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR":
        if cfg.lr_scheduler.T_max is None:
            cfg.lr_scheduler.T_max = num_iterations
    elif cfg.lr_scheduler._target_ == "torch.optim.lr_scheduler.OneCycleLR":
        if cfg.lr_scheduler.total_steps is None:
            cfg.lr_scheduler.total_steps = num_iterations
    scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer)

    amp_enabled = bool(cfg.amp.enabled and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    epoch_init = load_checkpoint(
        cfg.resume_dir,
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

    best_valid_loss = float("inf")
    best_loss_epoch = epoch_init
    logger.info("Starting training on %s", device)

    for epoch in range(epoch_init, int(cfg.train.epochs)):
        if manager.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        start = time.time()
        model.train()
        epoch_losses = {}

        for step, graph in enumerate(train_loader, start=1):
            losses = train_step(
                model,
                graph,
                dataset,
                criterion,
                optimizer,
                scaler,
                scheduler,
                device,
                amp_enabled,
            )
            for key, value in losses.items():
                epoch_losses.setdefault(key, []).append(value)
            if manager.rank == 0 and step % int(cfg.train.log_interval) == 0:
                logger.info(
                    "Epoch %s/%s step %s/%s loss %.4e acc_norm %.4e",
                    epoch + 1,
                    cfg.train.epochs,
                    step,
                    len(train_loader),
                    losses["loss"],
                    losses["loss_acc_norm"],
                )

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for graph in val_loader:
                graph = graph.to(device)
                gt_pos, _, gt_acc = datapipe.valid_dataset.unpack_targets(graph)
                pred_acc = model(graph.ndata["x"], graph.edata["x"], graph)
                mask = graph.ndata["mask"].unsqueeze(-1).to(pred_acc.dtype)
                num_nz = torch.clamp(mask.sum() * datapipe.valid_dataset.dim, min=1.0)
                loss_acc = (mask * criterion(pred_acc, gt_acc)).sum() / num_nz
                pos, vel, _ = datapipe.valid_dataset.unpack_inputs(graph)
                pred_pos, _ = datapipe.valid_dataset.time_integrator(
                    position=pos,
                    velocity=vel[-1],
                    acceleration=pred_acc,
                    dt=datapipe.valid_dataset.dt,
                    denormalize=True,
                )
                loss_pos = (mask * criterion(pred_pos, gt_pos)).sum() / num_nz
                valid_loss += (loss_acc + loss_pos).item()

        valid_loss /= max(len(val_loader), 1)

        if manager.rank == 0:
            mean_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            logger.info(
                "Epoch %s finished in %.2fs train_loss %.4e valid_loss %.4e lr %.2e",
                epoch + 1,
                time.time() - start,
                mean_losses.get("loss", 0.0),
                valid_loss,
                scheduler.get_last_lr()[0],
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_loss_epoch = epoch
                save_checkpoint(
                    cfg.resume_dir,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                )
                logger.info("Checkpoint saved to %s", cfg.resume_dir)
            if epoch - best_loss_epoch >= int(cfg.train.patience):
                logger.warning("Early stopping after %s stale epochs", cfg.train.patience)
                break

        if manager.distributed:
            torch.distributed.barrier()

    manager.cleanup()
    logger.info("Training finished")


if __name__ == "__main__":
    main()
