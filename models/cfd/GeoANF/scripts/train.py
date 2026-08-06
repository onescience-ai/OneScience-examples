"""Training script for GeoANF on the AirfRANS dataset."""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from omegaconf import OmegaConf
from onescience.datapipes.cfd import AirfRANSDatapipe

from model import GeoANF


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def geoanf_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    surf_mask: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute weighted MSE loss for surface and volume points.

    L = alpha * MSE_surf + MSE_vol
    """
    surf = surf_mask.bool().squeeze(-1) if surf_mask.dim() > 1 else surf_mask.bool()
    vol = ~surf

    total_loss = 0.0
    count = 0

    if surf.any():
        loss_surf = F.mse_loss(
            pred[:, surf, :], target[:, surf, :], reduction='mean'
        )
        total_loss += alpha * loss_surf
        count += 1

    if vol.any():
        loss_vol = F.mse_loss(
            pred[:, vol, :], target[:, vol, :], reduction='mean'
        )
        total_loss += loss_vol
        count += 1

    return total_loss / max(count, 1)


def train_epoch(
    model, dataloader, optimizer, device, alpha, sample_points
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # batch is a PyG Data object with .x, .y, .surf, .pos
        x = batch.x.to(device)     # [N, 7]
        y = batch.y.to(device)     # [N, 4]
        surf = batch.surf.to(device)  # [N, 1] or [N]

        # Separate boundary and query points
        surf_bool = surf.bool().squeeze(-1) if surf.dim() > 1 else surf.bool()
        boundary = x[surf_bool]    # boundary points
        n_total = x.shape[0]

        # Random subsampling
        if sample_points < n_total:
            idx = torch.randperm(n_total, device=device)[:sample_points]
            x_sample = x[idx]
            y_sample = y[idx]
            surf_sample = surf[idx]
        else:
            x_sample = x
            y_sample = y
            surf_sample = surf

        # Add batch dimension
        boundary = boundary.unsqueeze(0)      # (1, m, 7)
        x_sample = x_sample.unsqueeze(0)       # (1, n_sample, 7)
        y_sample = y_sample.unsqueeze(0)       # (1, n_sample, 4)
        surf_sample = surf_sample.unsqueeze(0)  # (1, n_sample, ...)

        optimizer.zero_grad()
        pred = model(boundary, x_sample)
        loss = geoanf_loss(pred, y_sample, surf_sample, alpha)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device, sample_points):
    model.eval()
    total_loss = 0.0
    total_surf_loss = 0.0
    total_vol_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch.x.to(device)
        y = batch.y.to(device)
        surf = batch.surf.to(device)

        surf_bool = surf.bool().squeeze(-1) if surf.dim() > 1 else surf.bool()
        boundary = x[surf_bool]
        n_total = x.shape[0]

        if sample_points < n_total:
            idx = torch.randperm(n_total, device=device)[:sample_points]
            x_sample = x[idx]
            y_sample = y[idx]
            surf_sample = surf[idx]
        else:
            x_sample = x
            y_sample = y
            surf_sample = surf

        boundary = boundary.unsqueeze(0)
        x_sample = x_sample.unsqueeze(0)
        y_sample = y_sample.unsqueeze(0)
        surf_sample = surf_sample.unsqueeze(0)

        pred = model(boundary, x_sample)
        loss = geoanf_loss(pred, y_sample, surf_sample, alpha=1.0)

        total_loss += loss.item()
        num_batches += 1

        # Track surface and volume separately
        surf_mask = surf_sample.bool().squeeze(-1) if surf_sample.dim() > 1 else surf_sample.bool()
        if surf_mask.any():
            total_surf_loss += F.mse_loss(
                pred[:, surf_mask, :], y_sample[:, surf_mask, :], reduction='mean'
            ).item()
        vol_mask = ~surf_mask
        if vol_mask.any():
            total_vol_loss += F.mse_loss(
                pred[:, vol_mask, :], y_sample[:, vol_mask, :], reduction='mean'
            ).item()

    n = max(num_batches, 1)
    return {
        'loss': total_loss / n,
        'surf_loss': total_surf_loss / n,
        'vol_loss': total_vol_loss / n,
    }


def main():
    parser = argparse.ArgumentParser(description='Train GeoANF on AirfRANS')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to AirfRANS data directory')
    parser.add_argument('--stats-dir', type=str, default=None,
                        help='Path to statistics directory (default: data-dir/stats)')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='Learning rate')
    parser.add_argument('--sample-points', type=int, default=16000,
                        help='Number of points to sample per simulation')
    parser.add_argument('--loss-alpha', type=float, default=1.0,
                        help='Weight for surface loss')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--train-split', type=str, default='full_train',
                        help='Training split name')
    parser.add_argument('--test-split', type=str, default='full_test',
                        help='Test split name')
    parser.add_argument('--val-split-ratio', type=float, default=0.1,
                        help='Validation split ratio from training set')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if args.stats_dir is None:
        args.stats_dir = os.path.join(args.data_dir, 'stats')

    os.makedirs(args.output_dir, exist_ok=True)

    # Build AirfRANS configuration
    cfg = OmegaConf.create({
        'datapipe': {
            'source': {
                'data_dir': args.data_dir,
                'stats_dir': args.stats_dir,
            },
            'data': {
                'splits': {
                    'train_name': args.train_split,
                    'test_name': args.test_split,
                    'val_split_ratio': args.val_split_ratio,
                },
                'sampling': {
                    'sample_strategy': None,
                },
                'subsampling': args.sample_points,
            },
            'model_hparams': {
                'build_graph': False,
            },
            'dataloader': {
                'batch_size': 1,
                'num_workers': 0,
                'pin_memory': True,
            },
        }
    })

    # Create dataloaders
    datapipe = AirfRANSDatapipe(cfg.datapipe, distributed=False)
    train_loader, _ = datapipe.train_dataloader()
    val_loader, _ = datapipe.val_dataloader()
    test_loader = datapipe.test_dataloader()

    print(f'Train batches: {len(train_loader.dataset)}, '
          f'Val batches: {len(val_loader.dataset)}, '
          f'Test batches: {len(test_loader.dataset)}')

    # Create model
    model = GeoANF(
        in_dim=7,
        hidden_dim=args.hidden_dim,
        out_dim=4,
        num_heads=args.num_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params}')

    optimizer = Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            args.loss_alpha, args.sample_points,
        )

        val_metrics = validate(model, val_loader, device, args.sample_points)

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_metrics["loss"]:.6f} | '
              f'Surf: {val_metrics["surf_loss"]:.6f} | '
              f'Vol: {val_metrics["vol_loss"]:.6f}')

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  -> Saved best model (val_loss={best_val_loss:.6f})')

    # Final test evaluation
    print('\nEvaluating on test set...')
    test_metrics = validate(model, test_loader, device, args.sample_points)
    print(f'Test Loss: {test_metrics["loss"]:.6f} | '
          f'Surf: {test_metrics["surf_loss"]:.6f} | '
          f'Vol: {test_metrics["vol_loss"]:.6f}')

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)

    # Save config
    config_path = os.path.join(args.output_dir, 'config.yaml')
    OmegaConf.save(OmegaConf.create({
        'model': {k: v for k, v in vars(args).items()
                   if k in ('hidden_dim', 'num_heads', 'sample_points', 'loss_alpha')},
        'training': {k: v for k, v in vars(args).items()
                      if k in ('epochs', 'lr', 'seed')},
        'results': {
            'best_val_loss': float(best_val_loss),
            'test_loss': float(test_metrics['loss']),
            'test_surf_loss': float(test_metrics['surf_loss']),
            'test_vol_loss': float(test_metrics['vol_loss']),
        },
    }), config_path)

    print(f'\nTraining complete. Output saved to {args.output_dir}')


if __name__ == '__main__':
    main()
