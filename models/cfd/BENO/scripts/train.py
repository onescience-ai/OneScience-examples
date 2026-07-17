import os
import random
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import HeteroGNS
from onescience.datapipes.cfd import BENODatapipe
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.utils.beno.utilities import LpLoss


def set_default_data_env():
    os.environ.setdefault("ONESCIENCE_BENO_DATA_DIR", str(PROJECT_ROOT / "data"))


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config():
    set_default_data_env()
    cfg = YParams(str(PROJECT_ROOT / "conf" / "config.yaml"), "root")
    cfg.datapipe.source.data_dir = str(resolve_path(cfg.datapipe.source.data_dir))
    cfg.datapipe.source.cache_dir = str(resolve_path(cfg.datapipe.source.cache_dir))
    cfg.training.output_dir = str(resolve_path(cfg.training.output_dir))
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def activation_from_name(name):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "leakyrelu":
        return nn.LeakyReLU
    return nn.SiLU


def build_model(model_cfg):
    return HeteroGNS(
        nnode_in_features=model_cfg.nnode_in_features,
        nnode_out_features=model_cfg.nnode_out_features,
        nedge_in_features=model_cfg.nedge_in_features,
        latent_dim=model_cfg.get("latent_dim", model_cfg.get("width", 128)),
        nmessage_passing_steps=model_cfg.get("nmessage_passing_steps", 10),
        nmlp_layers=model_cfg.nmlp_layers,
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim", model_cfg.get("width", 128)),
        activation=activation_from_name(model_cfg.act),
        boundary_dim=model_cfg.boundary_dim,
        trans_layer=model_cfg.trans_layer,
    )


def select_device(device_name, dist):
    device_name = str(device_name).lower()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device is cuda, but CUDA is not available.")
        return torch.device(f"cuda:{dist.local_rank}")
    return dist.device


def reduce_scalar(value, device, dist):
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if dist.world_size > 1:
        torch.distributed.all_reduce(tensor)
        tensor /= dist.world_size
    return tensor.item()


def evaluate(model, test_loader, device, u_normalizer, myloss, dist):
    model.eval()
    total_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = u_normalizer.decode(
                out.view(batch.num_graphs, -1),
                sample_idx=batch["G1"].sample_idx.view(batch.num_graphs, -1),
            )
            total_l2 += myloss(pred, batch["G1+2"].y.view(batch.num_graphs, -1)).item()
    total_l2 = reduce_scalar(total_l2, device, dist)
    return total_l2


def main():
    cfg = load_config()
    seed_everything(int(cfg.training.seed))

    DistributedManager.initialize()
    dist = DistributedManager()
    device = select_device(cfg.training.get("device", "auto"), dist)

    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Config: {PROJECT_ROOT / 'conf' / 'config.yaml'}")
        print(f"Data: {cfg.datapipe.source.data_dir}")
        print(f"Checkpoint directory: {output_dir}")

    datapipe = BENODatapipe(cfg, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    test_loader, _ = datapipe.test_dataloader()
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Check ntrain and batch_size.")

    u_normalizer = datapipe.u_normalizer.to(device)
    model = build_model(cfg.model).to(device)
    if dist.world_size > 1:
        device_ids = [dist.local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.training.scheduler.T_0,
        T_mult=cfg.training.scheduler.T_mult,
    )
    myloss = LpLoss(size_average=False)

    for epoch in range(int(cfg.training.epochs)):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        train_mse = 0.0
        train_l2 = 0.0
        batches = 0
        start = default_timer()
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.rank != 0))

        for batch in iterator:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = F.mse_loss(out.view(-1, 1), batch["G1+2"].y.view(-1, 1))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_denorm = u_normalizer.decode(
                    out.view(batch.num_graphs, -1),
                    sample_idx=batch["G1"].sample_idx.view(batch.num_graphs, -1),
                )
                target_denorm = u_normalizer.decode(
                    batch["G1+2"].y.view(batch.num_graphs, -1),
                    sample_idx=batch["G1"].sample_idx.view(batch.num_graphs, -1),
                )
                l2 = myloss(pred_denorm, target_denorm)

            train_mse += loss.item()
            train_l2 += l2.item()
            batches += 1
            if dist.rank == 0:
                iterator.set_postfix({"mse": f"{loss.item():.2e}", "l2": f"{l2.item():.2e}"})

        scheduler.step()
        train_mse = reduce_scalar(train_mse / batches, device, dist)
        train_l2 = reduce_scalar(train_l2 / datapipe.train_dataset.ntrain, device, dist)
        test_l2 = evaluate(model, test_loader, device, u_normalizer, myloss, dist)
        test_l2 /= datapipe.test_dataset.ntest

        if dist.rank == 0:
            print(
                f"Epoch {epoch:03d} | Train MSE: {train_mse:.6f} | "
                f"Train L2: {train_l2:.6f} | Test L2: {test_l2:.6f} | "
                f"Time: {default_timer() - start:.1f}s"
            )
            if (epoch + 1) % int(cfg.training.save_period) == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                checkpoint_name = cfg.training.checkpoint_name.format(epoch=epoch)
                ckpt_path = output_dir / checkpoint_name
                torch.save(model_to_save.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    dist.cleanup()


if __name__ == "__main__":
    main()
