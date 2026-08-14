"""enf2enf 完整模型（Section 3.2, Eq.2-8, Figure 1）。

G_theta = D_u o E_a。
- E_a：ENF 编码器 + CAVIA 元学习内循环（K 步梯度下降优化 latent features c_j）
- D_u：latent self-attention + 等价交叉注意力解码

latent positions p_j 固定（均匀覆盖 bounding box），不参与优化（Eq.7）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .enf import ENFEncoder, ENFDecoder


class ENF2ENF(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        n_lat: int = 9,
        l_dim: int = 8,
        width_enc: int = 128,
        width_dec: int = 256,
        heads: int = 2,
        sigma_window: float = 0.1,
        rff_d_enc: int = 128,
        rff_d_dec: int = 256,
        rff_sigma_enc: float = 2,
        rff_sigma_dec: float = 10,
        latent_bbox: tuple[float, float] = (-0.75, 0.75),
        attention_blocks: int = 2,
        inner_steps_K: int = 3,
        inner_lr_lambda_c: float = 0.5,
        geom_out_channels: int = 2,
        field_out_channels: int = 1,
        train_rff: bool = False,
        use_global_params: bool = False,
        global_param_dim: int = 0,
        share_input_decoder: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_lat = n_lat
        self.l_dim = l_dim
        self.inner_steps_K = inner_steps_K
        self.inner_lr_lambda_c = inner_lr_lambda_c
        self.use_global_params = use_global_params
        self.global_param_dim = global_param_dim
        self.seed = seed

        # latent 位置：均匀网格覆盖 bounding box，固定
        self.register_buffer(
            "latent_pos", self._uniform_latent_pos(latent_bbox, n_lat, coord_dim)
        )

        dec_l_dim = l_dim + global_param_dim if use_global_params else l_dim

        self.encoder = ENFEncoder(
            coord_dim=coord_dim,
            width=width_enc,
            heads=heads,
            n_lat=n_lat,
            l_dim=l_dim,
            sigma_window=sigma_window,
            rff_d=rff_d_enc,
            rff_sigma=rff_sigma_enc,
            out_channels=geom_out_channels,
            train_rff=train_rff,
            seed=seed,
        )
        if share_input_decoder:
            # 输入解码器 D_a 复用 encoder 网络（重构几何场）
            self.input_decoder = self.encoder
        else:
            self.input_decoder = ENFEncoder(
                coord_dim=coord_dim,
                width=width_enc,
                heads=heads,
                n_lat=n_lat,
                l_dim=l_dim,
                sigma_window=sigma_window,
                rff_d=rff_d_enc,
                rff_sigma=rff_sigma_enc,
                out_channels=geom_out_channels,
                train_rff=train_rff,
                seed=seed + 1,
            )
        self.decoder = ENFDecoder(
            coord_dim=coord_dim,
            width=width_dec,
            heads=heads,
            n_lat=n_lat,
            l_dim=dec_l_dim,
            sigma_window=sigma_window,
            rff_d=rff_d_dec,
            rff_sigma=rff_sigma_dec,
            attention_blocks=attention_blocks,
            out_channels=field_out_channels,
            train_rff=train_rff,
            seed=seed + 2,
        )

    def _uniform_latent_pos(self, bbox: tuple[float, float], n_lat: int, coord_dim: int) -> torch.Tensor:
        import math

        lo, hi = bbox
        if n_lat == 1:
            return torch.tensor([[(lo + hi) / 2] * coord_dim])
        grid = int(math.sqrt(n_lat))
        while grid * grid < n_lat:
            grid += 1
        xs = torch.linspace(lo, hi, grid)
        positions = []
        for i in range(grid):
            for j in range(grid):
                if len(positions) >= n_lat:
                    break
                if coord_dim == 2:
                    positions.append([xs[i].item(), xs[j].item()])
                else:
                    positions.append([xs[i].item()] + [(lo + hi) / 2] * (coord_dim - 1))
            if len(positions) >= n_lat:
                break
        return torch.tensor(positions, dtype=torch.float32)

    # ---------------- CAVIA 内循环 ----------------
    def _cavia_encode(self, x: torch.Tensor, a: torch.Tensor, c_init: torch.Tensor | None = None):
        """对 batch 中每个样本运行 K 步内循环优化 latent features c。

        x: (B, N, coord_dim) 查询坐标（归一化）
        a: (B, N, geom_out_channels) 几何场 target（归一化）
        返回优化后的 c: (B, n_lat, l_dim)
        """
        B = x.shape[0]
        p = self.latent_pos  # (n_lat,coord_dim)
        c = torch.zeros(B, self.n_lat, self.l_dim, device=x.device) if c_init is None else c_init

        # 冻结网络参数，仅优化 c（CAVIA 内循环始终启用 autograd）
        for p_ in self.encoder.parameters():
            p_.requires_grad_(False)

        try:
            with torch.enable_grad():
                c.requires_grad_(True)
                for _ in range(self.inner_steps_K):
                    pred = self.encoder(x, p, c)
                    loss = F.mse_loss(pred, a)
                    grads = torch.autograd.grad(loss, c, create_graph=False)[0]
                    c = (c - self.inner_lr_lambda_c * grads).detach()
                    c.requires_grad_(True)
        finally:
            # 恢复 encoder 参数可训练状态
            for p_ in self.encoder.parameters():
                p_.requires_grad_(True)

        c = c.detach()
        return c

    # ---------------- 前向 ----------------
    def forward_encoder(self, x: torch.Tensor, a: torch.Tensor):
        """输入编码：返回 z_a latent (B,n_lat,l_dim)。"""
        return self._cavia_encode(x, a)

    def encode(self, x: torch.Tensor, a: torch.Tensor):
        return self.forward_encoder(x, a)

    def decode(self, x: torch.Tensor, c: torch.Tensor, mu: torch.Tensor | None = None):
        """给定 latent c 与全局参数 mu，解码物理场。
        x:(B,N,coord_dim) c:(B,n_lat,l_dim) -> (B,N,field_out_channels)
        """
        p = self.latent_pos  # (n_lat,coord_dim)
        if self.use_global_params:
            assert mu is not None
            mu_b = mu.unsqueeze(1).expand(-1, self.n_lat, -1)  # (B,n_lat,mu_dim)
            c_in = torch.cat([c, mu_b], dim=-1)
        else:
            c_in = c
        return self.decoder(x, p, c_in)

    def reconstruct_geometry(self, x: torch.Tensor, c: torch.Tensor):
        """输入解码器 D_a 重建几何场（用于 L^a）。"""
        p = self.latent_pos  # (n_lat,coord_dim)
        return self.input_decoder(x, p, c)

    def forward(self, x: torch.Tensor, a: torch.Tensor, mu: torch.Tensor | None = None):
        """完整前向（训练联合使用）：编码 + 解码物理场。返回 (field_pred, latent, geom_pred)。"""
        c = self.forward_encoder(x, a)
        field = self.decode(x, c, mu)
        geom = self.reconstruct_geometry(x, c)
        return field, c, geom

    def infer(self, x: torch.Tensor, a: torch.Tensor, mu: torch.Tensor | None = None):
        """推理：CAVIA 编码 + 解码。x:(B,N,2) a:(B,N,2) -> (B,N,1)"""
        with torch.no_grad():
            c = self.forward_encoder(x, a)
            field = self.decode(x, c, mu)
        return field, c
