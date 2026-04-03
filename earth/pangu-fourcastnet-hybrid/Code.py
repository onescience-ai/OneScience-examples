import torch
import torch.nn as nn

from onescience.modules import (
    FourCastNetFuser,
    PanguEmbedding2D,
    PanguEmbedding3D,
    PanguPatchRecovery2D,
    PanguPatchRecovery3D,
)


class HybridPanguFourCastNetDay7(nn.Module):
    """
    用 Pangu 输入输出变量进行建模的 "7天一次性预测" 版本。

    语义：
    - 输入：t 时刻的 Pangu 变量（上空 5 个 + 地表 4 个 + 地表静态掩膜 3 个）
    - 输出：t+7天 的 Pangu 变量（上空 5 个 + 地表 4 个）

    结构（满足“全局传播 + 局地细节”的结合思路）：
    - 仍使用 Pangu 的 PatchEmbedding 与 PatchRecovery 保持 IO 变量、网格对齐与局部细化能力
    - 在中间特征层使用 FourCastNetFuser(AFN0) 做经纬面的全局频域混合（压力/层维度并入通道）
    """

    def __init__(
        self,
        # Pangu 默认分辨率：上空 13层、地表 721x1440
        upper_img_size=(13, 721, 1440),
        surface_img_size=(721, 1440),
        # Pangu 默认 patch_size：3D=(2,4,4)，2D=(4,4)
        pangu_patch_size_3d=(2, 4, 4),
        pangu_patch_size_2d=(4, 4),
        # Pangu 默认 embed_dim：192
        embed_dim=192,
        # AFNO 使用的 token 网格尺寸由 Pangu embedding 的输出决定：
        #   H' = ceil(721/4)=181, W' = ceil(1440/4)=360, D_total = 7(上空token层) + 1(地表token层) = 8
        afno_num_blocks=8,
        lead_time_days=7,
    ):
        super().__init__()
        self.lead_time_days = lead_time_days

        self.upper_img_size = tuple(upper_img_size)
        self.surface_img_size = tuple(surface_img_size)
        self.pangu_patch_size_3d = tuple(pangu_patch_size_3d)
        self.pangu_patch_size_2d = tuple(pangu_patch_size_2d)
        self.embed_dim = int(embed_dim)

        # Pangu IO 变量通道约定（按模块知识文件默认值）
        upper_in_chans = 5  # Z/Q/T/U/V
        surface_in_chans = 7  # MSLP/T2M/U10/V10 + 3个静态掩膜

        # 3D embedding token 网格维度
        level, lat, lon = self.upper_img_size
        pl_out = (level + self.pangu_patch_size_3d[0] - 1) // self.pangu_patch_size_3d[0]
        h_out = (lat + self.pangu_patch_size_3d[1] - 1) // self.pangu_patch_size_3d[1]
        w_out = (lon + self.pangu_patch_size_3d[2] - 1) // self.pangu_patch_size_3d[2]

        # 地表 embedding 输出会与上空的 (H', W') 对齐，形成总深度 D_total=pl_out + 1
        d_total = int(pl_out + 1)
        if (self.embed_dim * d_total) % afno_num_blocks != 0:
            raise ValueError(
                "AFNO dim 必须能被 num_blocks 整除："
                f"embed_dim={self.embed_dim}, d_total={d_total}, "
                f"dim={self.embed_dim * d_total}, num_blocks={afno_num_blocks}"
            )

        # Pangu embeddings
        self.upper_embed = PanguEmbedding3D(
            img_size=self.upper_img_size,
            patch_size=self.pangu_patch_size_3d,
            in_chans=upper_in_chans,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )
        self.surface_embed = PanguEmbedding2D(
            img_size=self.surface_img_size,
            patch_size=self.pangu_patch_size_2d,
            in_chans=surface_in_chans,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # AFNO 全局混合：对 [B, H', W', D_total*embed_dim] 做 2D FFT 混合
        self.afno_fuser = FourCastNetFuser(
            dim=self.embed_dim * d_total,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            double_skip=True,
            num_blocks=afno_num_blocks,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        )

        # Pangu recovery heads：将 AFNO 后的特征分别映射回上空(3D)与地表(2D)物理量
        self.upper_recovery = PanguPatchRecovery3D(
            img_size=self.upper_img_size,
            patch_size=self.pangu_patch_size_3d,
            in_chans=self.embed_dim,
            out_chans=5,
        )
        self.surface_recovery = PanguPatchRecovery2D(
            img_size=self.surface_img_size,
            patch_size=self.pangu_patch_size_2d,
            in_chans=self.embed_dim,
            out_chans=4,
        )

        # 用于中间 reshape 的一致性约束（避免潜在 off-by-one 错位）
        self._pl_out = int(pl_out)
        self._h_out = int(h_out)
        self._w_out = int(w_out)
        self._d_total = int(d_total)

    def forward(
        self,
        upper_air: torch.Tensor,
        surface_with_static: torch.Tensor,
    ):
        """
        Args:
            upper_air: [B, 5, 13, 721, 1440]，上空变量 (Z/Q/T/U/V)。
            surface_with_static: [B, 7, 721, 1440]，地表变量(4) + 静态掩膜(3)。

        Returns:
            upper_pred: [B, 5, 13, 721, 1440]，对应 t+7天。
            surface_pred: [B, 4, 721, 1440]，对应 t+7天。
        """
        if upper_air.ndim != 5:
            raise ValueError(f"upper_air expected 5D tensor, got shape={tuple(upper_air.shape)}")
        if surface_with_static.ndim != 4:
            raise ValueError(
                f"surface_with_static expected 4D tensor, got shape={tuple(surface_with_static.shape)}"
            )

        # (1) Embedding
        # upper_tokens: [B, embed_dim, pl_out, h_out, w_out]
        upper_tokens = self.upper_embed(upper_air)
        if (
            upper_tokens.shape[2] != self._pl_out
            or upper_tokens.shape[3] != self._h_out
            or upper_tokens.shape[4] != self._w_out
        ):
            raise RuntimeError(
                "Upper tokens shape mismatch with configured output grid: "
                f"got={tuple(upper_tokens.shape)}, expected_pl_out={self._pl_out}, "
                f"expected_h_out={self._h_out}, expected_w_out={self._w_out}"
            )

        # surface_tokens_2d: [B, embed_dim, h_out, w_out]
        surface_tokens_2d = self.surface_embed(surface_with_static)
        if surface_tokens_2d.shape[2] != self._h_out or surface_tokens_2d.shape[3] != self._w_out:
            raise RuntimeError(
                "Surface tokens shape mismatch with configured output grid: "
                f"got={tuple(surface_tokens_2d.shape)}, expected_h_out={self._h_out}, expected_w_out={self._w_out}"
            )

        # 将地表 token 视作 D_total 的最后一层：surface_tokens: [B, embed_dim, 1, h_out, w_out]
        surface_tokens = surface_tokens_2d.unsqueeze(2)

        # (2) 拼成 3D token 体：x_3d = [B, embed_dim, d_total, h_out, w_out]
        x_3d = torch.cat([upper_tokens, surface_tokens], dim=2)

        if x_3d.shape[2] != self._d_total:
            raise RuntimeError(
                f"Combined depth mismatch: got {x_3d.shape[2]} expected {self._d_total}"
            )

        # (3) AFNO 需要 [B, H, W, C]：把 d_total 并入通道
        # x_afno_in: [B, h_out, w_out, d_total*embed_dim]
        x_afno_in = x_3d.permute(0, 3, 4, 2, 1).contiguous()
        b, h, w, d, c = x_afno_in.shape
        x_afno_in = x_afno_in.view(b, h, w, d * c)

        # (4) 全局频域混合
        x_afno_out = self.afno_fuser(x_afno_in)  # [B, h_out, w_out, d_total*embed_dim]

        # (5) 恢复到 3D token 体： [B, embed_dim, d_total, h_out, w_out]
        x_afno_out = x_afno_out.view(b, h, w, self._d_total, self.embed_dim)
        x_3d_out = x_afno_out.permute(0, 4, 3, 1, 2).contiguous()

        # 上空恢复：取前 pl_out 层
        upper_feat = x_3d_out[:, :, : self._pl_out, :, :]
        upper_pred = self.upper_recovery(upper_feat)

        # 地表恢复：取最后 1 层
        surface_feat = x_3d_out[:, :, self._pl_out, :, :]
        surface_pred = self.surface_recovery(surface_feat)

        return upper_pred, surface_pred


def build_hybrid_pangu_fourcastnet_day7(**kwargs) -> HybridPanguFourCastNetDay7:
    """
    统一的构建入口，方便在外部训练/推理脚本中用配置驱动实例化。
    """

    return HybridPanguFourCastNetDay7(**kwargs)

