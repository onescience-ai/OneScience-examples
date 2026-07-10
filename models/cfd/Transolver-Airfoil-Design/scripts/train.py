from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import Transolver2D, Transolver2D_plus
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams


def resolve_path(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def setup_logging(rank: int) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("transolver-train")


def load_graph(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


class FakeAirfRANSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, stats_dir: Path, split: str, coef_norm=None):
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Fake manifest not found: {manifest_path}. Run scripts/fake_data.py first.")
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.data_dir = data_dir
        self.names = manifest[split]
        self.coef_norm = coef_norm or (
            np.load(stats_dir / "mean_in.npy"),
            np.load(stats_dir / "std_in.npy"),
            np.load(stats_dir / "mean_out.npy"),
            np.load(stats_dir / "std_out.npy"),
        )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        data = load_graph(self.data_dir / f"{self.names[index]}.pt")
        mean_in, std_in, mean_out, std_out = self.coef_norm
        data.x = (data.x - torch.as_tensor(mean_in, dtype=data.x.dtype)) / torch.as_tensor(std_in + 1e-8, dtype=data.x.dtype)
        data.y = (data.y - torch.as_tensor(mean_out, dtype=data.y.dtype)) / torch.as_tensor(std_out + 1e-8, dtype=data.y.dtype)
        return data


class FakeAirfRANSDatapipe:
    def __init__(self, cfg_data, distributed: bool):
        self.cfg_data = cfg_data
        self.distributed = distributed
        data_dir = resolve_path(cfg_data.source.data_dir)
        stats_dir = resolve_path(cfg_data.source.stats_dir)
        self.train_dataset = FakeAirfRANSDataset(data_dir, stats_dir, cfg_data.data.splits.train_name)
        self.coef_norm = self.train_dataset.coef_norm
        self.val_dataset = FakeAirfRANSDataset(data_dir, stats_dir, cfg_data.data.splits.val_name, self.coef_norm)
        self.test_dataset = FakeAirfRANSDataset(data_dir, stats_dir, cfg_data.data.splits.test_name, self.coef_norm)

    def _loader(self, dataset, shuffle: bool):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
        return (
            PyGDataLoader(
                dataset,
                batch_size=self.cfg_data.dataloader.batch_size,
                num_workers=self.cfg_data.dataloader.num_workers,
                pin_memory=torch.cuda.is_available(),
                shuffle=shuffle and sampler is None,
                sampler=sampler,
            ),
            sampler,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)


def build_model(cfg_model):
    model_name = cfg_model.name
    if model_name not in cfg_model.specific_params:
        raise ValueError(f"Model '{model_name}' is missing from model.specific_params")
    params = cfg_model.specific_params[model_name]
    model_cls = Transolver2D_plus if model_name == "Transolver_plus" else Transolver2D
    if model_name not in ("Transolver", "Transolver_plus"):
        raise NotImplementedError("This refactored package localizes only Transolver and Transolver_plus.")
    return model_cls(
        n_hidden=params.n_hidden,
        n_layers=params.n_layers,
        space_dim=params.space_dim,
        fun_dim=params.fun_dim,
        n_head=params.n_head,
        mlp_ratio=params.mlp_ratio,
        out_dim=params.out_dim,
        slice_num=params.slice_num,
        unified_pos=bool(params.unified_pos),
    )


def build_datapipe(cfg_data, model_params, distributed: bool):
    cfg_data.model_hparams = model_params
    if cfg_data.backend == "airfrans":
        from onescience.datapipes.cfd import AirfRANSDatapipe

        return AirfRANSDatapipe(params=cfg_data, distributed=distributed)
    if cfg_data.backend == "fake_airfrans":
        return FakeAirfRANSDatapipe(cfg_data, distributed=distributed)
    raise ValueError(f"Unsupported datapipe.backend: {cfg_data.backend}")


def masked_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return losses[mask].mean()
    return losses.mean() * 0.0


def save_checkpoint(model, optimizer, scheduler, epoch: int, loss: float, checkpoint_dir: Path, model_name: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        checkpoint_dir / f"{model_name}.pth",
    )


def main() -> int:
    config_path = ROOT / "conf" / "config.yaml"
    cfg_model = YParams(str(config_path), "model")
    cfg_data = YParams(str(config_path), "datapipe")
    cfg_train = YParams(str(config_path), "training")

    torch.manual_seed(int(cfg_train.seed))
    np.random.seed(int(cfg_train.seed))
    random.seed(int(cfg_train.seed))

    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)
    device_name = getattr(cfg_train, "device", "auto")
    if device_name == "cpu":
        device = torch.device("cpu")
    elif device_name.startswith("cuda"):
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{cfg_train.gpuid}" if torch.cuda.is_available() and manager.world_size == 1 else manager.device)

    model_params = cfg_model.specific_params[cfg_model.name]
    datapipe = build_datapipe(cfg_data, model_params, distributed=(manager.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("Train and validation loaders must both be non-empty.")

    model = build_model(cfg_model).to(device)
    if manager.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[manager.local_rank], output_device=manager.local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_train.lr))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(cfg_train.lr),
        total_steps=max(1, len(train_loader) * int(cfg_train.max_epoch)),
    )
    criterion = nn.L1Loss(reduction="none") if cfg_train.loss_criterion == "MAE" else nn.MSELoss(reduction="none")
    use_weighted_loss = cfg_train.loss_criterion == "MSE_weighted"
    checkpoint_dir = resolve_path(cfg_train.checkpoint_dir)
    best_valid_loss = float("inf")
    best_epoch = -1

    if manager.rank == 0:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Training %s with %.3fM parameters on %s", cfg_model.name, params / 1e6, device)

    for epoch in range(int(cfg_train.max_epoch)):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(data)
            losses = criterion(out, data.y)
            surf_loss = masked_mean(losses, data.surf)
            vol_loss = masked_mean(losses, ~data.surf)
            loss = vol_loss + float(cfg_train.loss_weight) * surf_loss if use_weighted_loss else losses.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += float(loss.detach().cpu())
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                losses = criterion(out, data.y)
                surf_loss = masked_mean(losses, data.surf)
                vol_loss = masked_mean(losses, ~data.surf)
                loss = vol_loss + float(cfg_train.loss_weight) * surf_loss if use_weighted_loss else losses.mean()
                if manager.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                valid_loss += float(loss.detach().cpu())
        valid_loss /= len(val_loader)

        if manager.rank == 0:
            logger.info(
                "Epoch %d/%d train=%.6f valid=%.6f time=%.2fs",
                epoch + 1,
                int(cfg_train.max_epoch),
                train_loss,
                valid_loss,
                time.time() - start,
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                save_checkpoint(model, optimizer, scheduler, epoch, valid_loss, checkpoint_dir, cfg_model.name)
                logger.info("Saved checkpoint to %s", checkpoint_dir / f"{cfg_model.name}.pth")
            if epoch - best_epoch > int(cfg_train.patience):
                logger.info("Early stopping after %d epochs without improvement.", int(cfg_train.patience))
                break

    if manager.world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
