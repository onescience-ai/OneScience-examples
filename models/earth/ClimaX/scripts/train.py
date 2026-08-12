import torch
import os
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
import shutil
import numpy as np
import torch.distributed as dist
import logging
import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from model.ClimaX import ClimaX
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams


# ============================================================================
# Loss function: Latitude-weighted MSE (from official ClimaX metrics.py)
# ============================================================================

def lat_weighted_mse(pred, y, lat):
    """Latitude weighted mean squared error.

    Allows to weight the loss by the cosine of the latitude to account for
    gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        lat: [H] latitude array in degrees

    Returns:
        scalar loss
    """
    error = (pred - y) ** 2  # [B, V, H, W]

    # latitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H,)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(
        dtype=error.dtype, device=error.device
    )  # (1, H, 1)

    loss = (error * w_lat.unsqueeze(1)).mean()
    return loss


def get_lat_array(img_size, spatial_res=5.625):
    """Generate latitude array for the grid.

    Args:
        img_size: [H, W]
        spatial_res: degrees per grid cell

    Returns:
        lat: [H] latitude values from north to south
    """
    H = img_size[0]
    # Cell centers: from 90 - res/2 to -90 + res/2
    lat = np.linspace(90 - spatial_res / 2, -90 + spatial_res / 2, H)
    return lat.astype(np.float32)


# ============================================================================
# Training
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")

    ## Distributed config init
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    world_rank = 0
    local_rank = 0
    if cfg.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()

    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    # Build variable lists from config
    all_vars = cfg_data.dataset.channels
    out_vars = cfg_data.dataset.out_variables

    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=all_vars,
        used_years=cfg_data.dataset.train_time,
        distributed=dist.is_initialized(),
    )
    train_dataloader, train_sampler = datapipe.get_dataloader("train")

    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=all_vars,
        used_years=cfg_data.dataset.val_time,
        distributed=dist.is_initialized(),
    )
    val_dataloader, val_sampler = datapipe.get_dataloader("valid")

    ## Model init
    model = ClimaX(
        default_vars=all_vars,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        decoder_depth=cfg.decoder_depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        drop_path=cfg.drop_path,
        drop_rate=cfg.drop_rate,
    ).to(local_rank)

    ## Optimizer (following official ClimaX: AdamW with param groups)
    decay = []
    no_decay = []
    for name, m in model.named_parameters():
        if "var_embed" in name or "pos_embed" in name:
            no_decay.append(m)
        else:
            decay.append(m)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "lr": cfg.lr,
                "betas": (cfg.beta_1, cfg.beta_2),
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": no_decay,
                "lr": cfg.lr,
                "betas": (cfg.beta_1, cfg.beta_2),
                "weight_decay": 0,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=5, mode="min"
    )

    ## Get latitude array for lat-weighted loss
    lat = get_lat_array(cfg.img_size)

    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    ## Get model params count
    if cfg.world_size == 1 or world_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n\n")
        print("-" * 50)
        print(f"Model params: {total_params}, {total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B")
        print("-" * 50, "\n")

    ## Load model weight if there exists a well-trained model
    if os.path.exists(f"{cfg.checkpoint_dir}/model_bak.pth"):
        if world_rank == 0:
            print("\n\n")
            print("-" * 50)
            print(f"Found existing model weight, loading and continuing training...")
            print(f"If you want to train a new model, remove *.pth from {cfg.checkpoint_dir}")
            print("-" * 50, "\n")
        ckpt = torch.load(
            f"{cfg.checkpoint_dir}/model_bak.pth",
            map_location=f'cuda:{local_rank}',
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_valid_loss = ckpt["best_valid_loss"]
        best_loss_epoch = ckpt["best_loss_epoch"]
        train_losses = np.load(train_loss_file)
        valid_losses = np.load(valid_loss_file)

    ## Pre-compute output variable indices (before DDP wrap, for loss computation)
    out_var_indices = model.get_var_ids(tuple(out_vars), torch.device('cpu'))

    ## Distributed model
    if cfg.world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    ## Lead time (fixed for deterministic forecasting)
    predict_range = cfg.predict_range
    hrs_each_step = cfg.hrs_each_step
    lead_time_val = (predict_range * hrs_each_step) / 100.0  # normalized

    world_rank == 0 and logger.info(f"Starting training... lead_time={lead_time_val}")

    for epoch in range(cfg.max_epoch):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0
        start_time = time.time()

        for j, data in enumerate(train_dataloader):
            invar = data[0].to(local_rank, dtype=torch.float32)   # [B, C_all, H, W]
            outvar = data[1].to(local_rank, dtype=torch.float32)  # [B, C_all, H, W]

            preds = model(invar, all_vars, out_vars, lead_time_val)

            # Select output variable channels from ground truth
            outvar_selected = outvar[:, out_var_indices.to(outvar.device)]
            loss = lat_weighted_mse(preds, outvar_selected, lat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if world_rank == 0:
                logger.info(
                    f'Train: Epoch {epoch}-{j+1}/{len(train_dataloader)} '
                    f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                    f'[{(time.time()-start_time)/(j+1): .02f}s/batch] '
                    f'loss:{train_loss / (j+1): .04f}'
                )

        train_loss /= len(train_dataloader)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for j, data in enumerate(val_dataloader):
                invar = data[0].to(local_rank, dtype=torch.float32)
                outvar = data[1].to(local_rank, dtype=torch.float32)

                preds = model(invar, all_vars, out_vars, lead_time_val)

                # Select output variable channels from ground truth
                outvar_selected = outvar[:, out_var_indices.to(outvar.device)]
                loss = lat_weighted_mse(preds, outvar_selected, lat)

                if cfg.world_size > 1:
                    loss_tensor = loss.detach().to(local_rank)
                    dist.all_reduce(loss_tensor)
                    valid_loss += loss_tensor.item() / cfg.world_size
                else:
                    valid_loss += loss.item()

                if world_rank == 0:
                    logger.info(
                        f'Valid: Epoch {epoch}-{j+1}/{len(val_dataloader)} '
                        f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                        f'[{(time.time()-start_time)/(j+1): .02f}s/batch] '
                        f'loss:{valid_loss / (j+1): .04f}'
                    )

        valid_loss /= len(val_dataloader)
        is_save_ckp = False

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            world_rank == 0 and save_checkpoint(
                model, optimizer, scheduler,
                best_valid_loss, best_loss_epoch, cfg.checkpoint_dir
            )
            is_save_ckp = True

        scheduler.step(valid_loss)

        if world_rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{cfg.max_epoch}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}, "
                f"Best loss at Epoch: {best_loss_epoch + 1}"
                + (", saving checkpoint" if is_save_ckp else "")
            )
            train_losses = np.append(train_losses, train_loss)
            valid_losses = np.append(valid_losses, valid_loss)

            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)

        if epoch - best_loss_epoch > cfg.patience:
            print(f"Loss has not decreased in {cfg.patience} epochs, stopping training...")
            sys.exit()


def save_checkpoint(model, optimizer, scheduler, best_valid_loss,
                    best_loss_epoch, model_path):
    # Only save on global rank 0 (guard against misconfigured multi-process launch)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
    }
    # Write to temporary file first, then atomically rename to avoid
    # race conditions from misconfigured multi-process launches
    tmp_path = f"{model_path}/model_tmp.pth"
    dst_path = f"{model_path}/model_bak.pth"
    torch.save(state, tmp_path)
    shutil.move(tmp_path, dst_path)


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
