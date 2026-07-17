from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check(name: str, tensor: torch.Tensor, expected_last_dim: int | None = None) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} produced non-finite values")
    if expected_last_dim is not None and tensor.shape[-1] != expected_last_dim:
        raise RuntimeError(
            f"{name} last dim expected {expected_last_dim}, got {tensor.shape[-1]}"
        )
    print(f"[OK] {name}: shape={tuple(tensor.shape)}")


@torch.no_grad()
def smoke_deeponet() -> None:
    from model.deeponet import DeepONetCartesianProd1D

    model = DeepONetCartesianProd1D(
        size=8,
        in_channel_branch=1,
        query_dim=1,
        out_channel=1,
        activation="relu",
        base_model="MLP",
    ).eval()
    x_func = torch.randn(2, 8, 1)
    x_loc = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
    out = model((x_func, x_loc))
    check("deeponet.DeepONetCartesianProd1D", out, expected_last_dim=1)


@torch.no_grad()
def smoke_fno() -> None:
    from model.fno import FNO2d

    model = FNO2d(num_channels=1, modes1=4, modes2=4, width=8, initial_step=1).eval()
    x = torch.randn(2, 16, 16, 1)
    coords = torch.linspace(0.0, 1.0, 16)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(2, 1, 1, 1)
    out = model(x, grid)
    check("fno.FNO2d", out, expected_last_dim=1)


@torch.no_grad()
def smoke_mpnn() -> None:
    from torch_geometric.data import Data

    from model.mpnn import MPNN

    pde = SimpleNamespace(
        name="smoke",
        tmin=0.0,
        tmax=1.0,
        resolution_t=10,
        spatial_domain=[(0.0, 1.0)],
        resolution=[4],
        spatial_dim=1,
        variables={},
    )
    model = MPNN(pde=pde, time_window=10, hidden_features=128, hidden_layers=1).eval()
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 2, 3]],
        dtype=torch.long,
    )
    data = Data(
        x=torch.randn(4, 10, 1),
        x_pos=torch.linspace(0.0, 1.0, 4).unsqueeze(-1),
        t_pos=torch.zeros(4),
        edge_index=edge_index,
        batch=torch.zeros(4, dtype=torch.long),
        variables=torch.empty(4, 0),
    )
    out = model(data, v=0)
    check("mpnn.MPNN", out)


@torch.no_grad()
def smoke_pino_fno() -> None:
    from model.pino_fno import FNO1d

    model = FNO1d(
        modes=[4, 4],
        width=4,
        layers=[4, 4, 4],
        fc_dim=8,
        in_dim=2,
        out_dim=1,
        pad_ratio=[0.0, 0.0],
    ).eval()
    out = model(torch.randn(2, 16, 2))
    check("pino_fno.FNO1d", out, expected_last_dim=1)


@torch.no_grad()
def smoke_unet() -> None:
    from model.unet import UNet2d

    model = UNet2d(in_channels=1, out_channels=1, init_features=2).eval()
    out = model(torch.randn(1, 1, 16, 16))
    check("unet.UNet2d", out)


@torch.no_grad()
def smoke_uno() -> None:
    from model.uno import UNO1d

    model = UNO1d(num_channels=1, width=4, initial_step=1).eval()
    x = torch.randn(1, 64, 1)
    grid = torch.linspace(0.0, 1.0, 64).view(1, 64, 1)
    out = model(x, grid)
    check("uno.UNO1d", out, expected_last_dim=1)


def main() -> int:
    torch.manual_seed(0)
    smoke_deeponet()
    smoke_fno()
    smoke_mpnn()
    smoke_pino_fno()
    smoke_unet()
    smoke_uno()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
