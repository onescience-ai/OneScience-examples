import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.meshgraphnet import MeshGraphNet
from onescience.distributed import DistributedManager
from onescience.utils.YParams import YParams
from onescience.launch.utils import load_checkpoint, save_checkpoint
from fake_data import build_cylinder_flow_datapipe


def setup_logging(rank: int):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("mesh_graph_net.train")


def build_model(model_params, device):
    mlp_act = "silu" if model_params.recompute_activation else "relu"
    return MeshGraphNet(
        input_dim_nodes=model_params.num_input_features,
        input_dim_edges=model_params.num_edge_features,
        output_dim=model_params.num_output_features,
        processor_size=model_params.processor_size,
        hidden_dim_processor=model_params.hidden_dim_processor,
        num_layers_node_processor=model_params.num_layers_node_processor,
        num_layers_edge_processor=model_params.num_layers_edge_processor,
        hidden_dim_node_encoder=model_params.hidden_dim_node_encoder,
        hidden_dim_edge_encoder=model_params.hidden_dim_edge_encoder,
        hidden_dim_node_decoder=model_params.hidden_dim_node_decoder,
        mlp_activation_fn=mlp_act,
        do_concat_trick=model_params.do_concat_trick,
        num_processor_checkpoint_segments=model_params.num_processor_checkpoint_segments,
        recompute_activation=model_params.recompute_activation,
    ).to(device)


def graph_from_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def resolve_device(device_name: str, manager: DistributedManager, gpuid: int):
    if manager.world_size > 1:
        return manager.device
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested cuda device, but torch.cuda.is_available() is false.")
        return torch.device(f"cuda:{gpuid}")
    return torch.device(f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu")


def main():
    os.chdir(PROJECT_ROOT)
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg_model = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")
    cfg_train = YParams(config_path, "training")
    model_params = cfg_model.specific_params[cfg_model.name]

    datapipe = build_cylinder_flow_datapipe(
        params=cfg_data,
        distributed=(manager.world_size > 1),
        project_root=PROJECT_ROOT,
    )
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()

    device = resolve_device(getattr(cfg_train, "device", "auto"), manager, cfg_train.gpuid)
    logger.info("Using device: %s", device)
    model = build_model(model_params, device)
    if manager.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[manager.local_rank], output_device=manager.local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cfg_train.lr_decay_rate**step)
    loss_criterion = nn.MSELoss() if cfg_train.loss_criterion == "MSE" else nn.L1Loss()
    scaler = GradScaler(enabled=bool(cfg_train.amp))

    checkpoint_dir = PROJECT_ROOT / cfg_train.checkpoint_dir
    epoch_init = load_checkpoint(checkpoint_dir, models=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device)
    best_valid_loss = float("inf")
    best_loss_epoch = epoch_init

    logger.info("Starting training")
    for epoch in range(epoch_init, cfg_train.max_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        start = time.time()
        model.train()
        train_loss = 0.0
        for idx, batch in enumerate(train_loader):
            graph = graph_from_batch(batch).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=bool(cfg_train.amp)):
                pred = model(graph.ndata["x"], graph.edata["x"], graph)
                loss = loss_criterion(pred, graph.ndata["y"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            if manager.rank == 0 and (idx + 1) % cfg_train.log_interval == 0:
                logger.info("Epoch %s/%s batch %s/%s loss %.6f", epoch + 1, cfg_train.max_epoch, idx + 1, len(train_loader), loss.item())

        train_loss /= max(len(train_loader), 1)
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                graph = graph_from_batch(batch).to(device)
                with autocast(device_type=device.type, enabled=bool(cfg_train.amp)):
                    pred = model(graph.ndata["x"], graph.edata["x"], graph)
                    loss = loss_criterion(pred, graph.ndata["y"])
                valid_loss += loss.item()
        valid_loss /= max(len(val_loader), 1)

        if manager.rank == 0:
            logger.info(
                "Epoch %s finished in %.2fs train_loss %.6f valid_loss %.6f",
                epoch + 1,
                time.time() - start,
                train_loss,
                valid_loss,
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_loss_epoch = epoch
                save_checkpoint(checkpoint_dir, models=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch + 1)
                logger.info("Checkpoint saved to %s", checkpoint_dir)
            if (epoch - best_loss_epoch) > cfg_train.patience:
                logger.warning("Early stopping after %s stale epochs", cfg_train.patience)
                break

    logger.info("Training finished")


if __name__ == "__main__":
    main()
