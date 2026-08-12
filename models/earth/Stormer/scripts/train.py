"""
Stormer Training Script — Matching Official Implementation.

Training follows the official Stormer approach exactly:
1. Input x(t) is normalized with inp_transform (N(0,1) per variable)
2. Model predicts Δx in DIFF-NORMALIZED space
3. Ground truth: Δx_gt = diff_transform[Δt](raw_out - raw_in)
4. Loss: L1 between pred_diff and gt_diff (both in diff-normalized space)
5. Autoregressive rollout with: norm_diff → raw_diff → original → re-normalize

Supports:
- Single GPU:     python scripts/train.py
- Multi-GPU:      torchrun --nproc_per_node=N scripts/train.py
- Cluster (slurm): sbatch work_slurm.sh
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings from external libraries (apex, etc.)
warnings.filterwarnings("ignore", category=UserWarning, module="apex")
warnings.filterwarnings("ignore", message=".*DtypeTensor constructors.*")

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import numpy as np
import logging
import time
import random

from torch.nn.parallel import DistributedDataParallel
from model.stormer import Stormer
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams

try:
    from apex import optimizers
    HAS_APEX = True
except ImportError:
    HAS_APEX = False


# ============================================================================
# Normalization utilities
# ============================================================================

class Normalize:
    """Per-variable normalization: y = (x - mean) / std.

    Replaces torchvision.transforms.Normalize to avoid dependency.
    """

    def __init__(self, mean, std):
        # mean, std: (V,) tensors
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

    def __call__(self, x):
        # x: (B, V, H, W) or (V, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            return ((x - self.mean) / self.std).squeeze(0)
        return (x - self.mean) / self.std


def get_reverse_transform(transform):
    """Return the inverse of a Normalize transform."""
    mean = transform.mean.view(-1)
    std = transform.std.view(-1)
    std_rev = 1.0 / std
    mean_rev = -mean * std_rev
    return Normalize(mean_rev, std_rev)


def load_normalization_stats(normalize_dir, variables):
    """Load official Stormer normalization constants.

    Returns:
        inp_transform: Normalize for input fields
        reverse_inp_transform: inverse of inp_transform
        diff_transform: dict {interval: Normalize} for diff fields
        reverse_diff_transform: dict {interval: Normalize} inverse
    """
    # Input normalization
    mean_dict = dict(np.load(os.path.join(normalize_dir, "normalize_mean.npz")))
    std_dict = dict(np.load(os.path.join(normalize_dir, "normalize_std.npz")))

    inp_mean = np.concatenate([mean_dict[v] for v in variables], axis=0)
    inp_std = np.concatenate([std_dict[v] for v in variables], axis=0)

    inp_mean_t = torch.from_numpy(inp_mean).float()
    inp_std_t = torch.from_numpy(inp_std).float()

    inp_transform = Normalize(inp_mean_t, inp_std_t)
    reverse_inp_transform = get_reverse_transform(inp_transform)

    # Diff normalization for each interval
    diff_transform = {}
    reverse_diff_transform = {}
    for interval in [6, 12, 24]:
        dmean_dict = dict(np.load(
            os.path.join(normalize_dir, f"normalize_diff_mean_{interval}.npz")))
        dstd_dict = dict(np.load(
            os.path.join(normalize_dir, f"normalize_diff_std_{interval}.npz")))

        dmean = np.concatenate([dmean_dict[v] for v in variables], axis=0)
        dstd = np.concatenate([dstd_dict[v] for v in variables], axis=0)

        dmean_t = torch.from_numpy(dmean).float()
        dstd_t = torch.from_numpy(dstd).float()

        diff_transform[interval] = Normalize(dmean_t, dstd_t)
        reverse_diff_transform[interval] = get_reverse_transform(
            diff_transform[interval])

    return (inp_transform, reverse_inp_transform,
            diff_transform, reverse_diff_transform)


# ============================================================================
# Training
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    # ============================================================
    # Config
    # ============================================================
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    # ============================================================
    # Distributed setup
    # ============================================================
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    world_rank = 0
    local_rank = 0
    if cfg.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()

    # ============================================================
    # Load normalization stats
    # ============================================================
    normalize_dir = cfg.normalize_dir
    variables = cfg_data.dataset.channels

    (inp_transform, reverse_inp_transform,
     diff_transform, reverse_diff_transform) = load_normalization_stats(
         normalize_dir, variables)

    # Move transforms to device
    for key in diff_transform:
        diff_transform[key].mean = diff_transform[key].mean.to(local_rank)
        diff_transform[key].std = diff_transform[key].std.to(local_rank)
        reverse_diff_transform[key].mean = reverse_diff_transform[key].mean.to(local_rank)
        reverse_diff_transform[key].std = reverse_diff_transform[key].std.to(local_rank)
    inp_transform.mean = inp_transform.mean.to(local_rank)
    inp_transform.std = inp_transform.std.to(local_rank)
    reverse_inp_transform.mean = reverse_inp_transform.mean.to(local_rank)
    reverse_inp_transform.std = reverse_inp_transform.std.to(local_rank)

    # ============================================================
    # DataLoader — get RAW data (normalize=False since we apply
    # official normalization manually)
    # ============================================================
    max_output_steps = max(cfg.list_train_intervals) // cfg.data_freq  # 24/6 = 4

    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=variables,
        used_years=cfg_data.dataset.train_time,
        distributed=dist.is_initialized(),
        input_steps=1,
        output_steps=max_output_steps,
        normalize=False,  # Raw data — apply official normalization manually
        batch_size=cfg_data.dataloader.batch_size,
        num_workers=cfg_data.dataloader.num_workers,
    )
    train_dataloader, train_sampler = datapipe.get_dataloader("train")

    # Validation: use output_steps=1 (6h) to avoid negative samples_per_year
    # with T=10 fake data (10-1-12+1 < 0 for output_steps=12)
    val_datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=variables,
        used_years=cfg_data.dataset.val_time,
        distributed=dist.is_initialized(),
        input_steps=1,
        output_steps=1,
        normalize=False,
        batch_size=cfg_data.dataloader.batch_size,
        num_workers=cfg_data.dataloader.num_workers,
    )
    val_dataloader, val_sampler = val_datapipe.get_dataloader("valid")

    # ============================================================
    # Model
    # ============================================================
    model = Stormer(
        in_img_size=cfg.in_img_size,
        variables=variables,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
    ).to(local_rank)

    # Mixed precision: FP16 autocast + GradScaler (matches official precision=16)
    use_amp = (local_rank >= 0)  # enable AMP when GPU/DCU available
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Optimizer — matching official: separate weight decay for embedding params
    decay = []
    no_decay = []
    for name, m in model.named_parameters():
        if "channel_embed" in name or "pos_embed" in name:
            no_decay.append(m)
        else:
            decay.append(m)

    if HAS_APEX:
        optimizer = optimizers.FusedAdam(
            [{"params": decay, "lr": cfg.lr,
              "betas": (cfg.beta_1, cfg.beta_2),
              "weight_decay": cfg.weight_decay},
             {"params": no_decay, "lr": cfg.lr,
              "betas": (cfg.beta_1, cfg.beta_2),
              "weight_decay": 0}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [{"params": decay, "lr": cfg.lr,
              "betas": (cfg.beta_1, cfg.beta_2),
              "weight_decay": cfg.weight_decay},
             {"params": no_decay, "lr": cfg.lr,
              "betas": (cfg.beta_1, cfg.beta_2),
              "weight_decay": 0}]
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=5, mode="min"
    )

    # Latitude-weighted L1 loss
    lat_path = os.path.join(cfg_data.dataset.data_dir, "static", "lat.npy")
    if os.path.exists(lat_path):
        lat = np.load(lat_path)
    else:
        lat = np.linspace(90, -90, cfg.in_img_size[0])
    loss_obj = _LatWeightedL1Loss(lat, local_rank)

    # ============================================================
    # Training state
    # ============================================================
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    if world_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n" + "-" * 50)
        print(f"📂 Stormer params: {total_params:,} "
              f"({total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B)")
        print(f"   Resolution: {cfg.in_img_size}")
        print(f"   Patch size: {cfg.patch_size}, Hidden: {cfg.hidden_size}")
        print(f"   Depth: {cfg.depth}, Heads: {cfg.num_heads}")
        print(f"   Variables: {len(variables)}")
        print(f"   Intervals: {cfg.list_train_intervals}")
        print(f"   Norm dir: {normalize_dir}")
        print("-" * 50 + "\n")

    # Load checkpoint
    if os.path.exists(f"{cfg.checkpoint_dir}/model_bak.pth"):
        if world_rank == 0:
            print(f"✅ Found checkpoint, resuming training...")
        ckpt = torch.load(
            f"{cfg.checkpoint_dir}/model_bak.pth",
            map_location=f'cuda:{local_rank}', weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_valid_loss = ckpt["best_valid_loss"]
        best_loss_epoch = ckpt["best_loss_epoch"]
        if os.path.exists(train_loss_file):
            train_losses = np.load(train_loss_file)
        if os.path.exists(valid_loss_file):
            valid_losses = np.load(valid_loss_file)

    # DDP
    if cfg.world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    if world_rank == 0:
        logger.info("Starting Stormer training...")

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        train_loss = 0
        start_time = time.time()

        for j, data in enumerate(train_dataloader):
            # ERA5Dataset (normalize=False):
            #   DataLoader adds batch dim → need to squeeze(batch_size=1)
            #   invar: (B, C, H, W) → squeeze → (C, H, W) raw at time t
            #   outvar: (B, T_out, C, H, W) → squeeze → (T_out, C, H, W) raw future
            invar = data[0].to(local_rank, dtype=torch.float32).squeeze(0)
            outvar = data[1].to(local_rank, dtype=torch.float32).squeeze(0)

            # Randomly select training interval
            chosen_interval = random.choice(cfg.list_train_intervals)
            step_idx = chosen_interval // cfg.data_freq - 1  # 6→0, 12→1, 24→3

            # ---- Forward pass following official forward_train logic ----
            # Step 1: Normalize input: (C,H,W) → (V,H,W) → unsqueeze → (1,V,H,W)
            x = inp_transform(invar).unsqueeze(0)  # (1, V, H, W)

            # Step 2: Compute ground truth diff in DIFF-NORMALIZED space
            # raw_diff = outvar[step_idx] - invar (both raw, step_idx on time dim)
            raw_diff = outvar[step_idx] - invar  # (V, H, W)
            gt_norm_diff = diff_transform[chosen_interval](raw_diff)  # (V, H, W)

            # Step 3: Model forward with AMP + gradient checkpointing
            interval_tensor = torch.tensor(
                [chosen_interval], device=local_rank, dtype=torch.float32
            )
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                pred_norm_diff = model(
                    x, variables, interval_tensor, use_checkpoint=True,
                )  # (1, V, H, W)
            pred_norm_diff = _replace_constant(pred_norm_diff.float(), variables)

            # Step 4: Loss in diff-normalized space
            gt_norm_diff = gt_norm_diff.unsqueeze(0)  # (1, V, H, W)
            loss = loss_obj(pred_norm_diff, gt_norm_diff)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if world_rank == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f'Train: Epoch {epoch}-{j+1}/{len(train_dataloader)} '
                    f'[cost {int(elapsed // 60):02}:{int(elapsed % 60):02}] '
                    f'[{elapsed/(j+1):.02f}s/batch] '
                    f'interval={chosen_interval}h '
                    f'loss:{train_loss / (j+1):.04f}'
                )

        train_loss /= len(train_dataloader)

        # ---- Validation ----
        model.eval()
        valid_loss = 0
        val_start = time.time()

        with torch.no_grad():
            for j, data in enumerate(val_dataloader):
                # Squeeze batch dim (batch_size=1)
                invar = data[0].to(local_rank, dtype=torch.float32).squeeze(0)
                outvar = data[1].to(local_rank, dtype=torch.float32).squeeze(0)

                # Use 6h interval for validation (simple next-step)
                val_interval = 6
                target_frame = outvar[0]  # t+6h

                # Same forward logic as training
                x = inp_transform(invar).unsqueeze(0)
                raw_diff = target_frame - invar
                gt_norm_diff = diff_transform[val_interval](raw_diff)

                interval_tensor = torch.tensor(
                    [val_interval], device=local_rank, dtype=torch.float32
                )
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                    pred_norm_diff = model(
                        x, variables, interval_tensor, use_checkpoint=False,
                    )
                pred_norm_diff = _replace_constant(pred_norm_diff.float(), variables)

                gt_norm_diff = gt_norm_diff.unsqueeze(0)
                loss = loss_obj(pred_norm_diff, gt_norm_diff)

                if cfg.world_size > 1:
                    loss_tensor = loss.detach().to(local_rank)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                    valid_loss += loss
                else:
                    valid_loss += loss.item()

                if world_rank == 0:
                    logger.info(
                        f'Valid: Epoch {epoch}-{j+1}/{len(val_dataloader)} '
                        f'[{(time.time()-val_start)/(j+1):.02f}s/batch] '
                        f'loss:{valid_loss / (j+1):.04f}'
                    )

        valid_loss /= len(val_dataloader)

        # ---- Checkpoint & Early stopping ----
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            if world_rank == 0:
                save_checkpoint(
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
            if world_rank == 0:
                print(f"Loss has not decreased in {cfg.patience} epochs, stopping.")
            break


# ============================================================================
# Helper: zero out constant variable predictions
# ============================================================================

from model.stormer import CONSTANTS

def _replace_constant(yhat, out_variables):
    """Zero out diffs for constant/invariant variables."""
    for i in range(yhat.shape[1]):
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = 0.0
    return yhat


# ============================================================================
# Latitude-weighted L1 Loss
# ============================================================================

class _LatWeightedL1Loss(nn.Module):
    """Latitude-weighted L1 loss."""

    def __init__(self, lat, device):
        super().__init__()
        w_lat = np.cos(np.deg2rad(lat))
        w_lat = w_lat / w_lat.mean()
        self.w_lat = torch.from_numpy(w_lat).float().to(device).unsqueeze(0).unsqueeze(-1)

    def forward(self, pred, target):
        error = torch.abs(pred - target)  # (B, V, H, W)
        weighted = error * self.w_lat.unsqueeze(1)  # (B, V, H, W)
        return weighted.mean()


# ============================================================================
# Checkpoint utilities
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, best_valid_loss,
                    best_loss_epoch, model_path):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
    }
    torch.save(state, f"{model_path}/model.pth")
    os.system(f"cp {model_path}/model.pth {model_path}/model_bak.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
