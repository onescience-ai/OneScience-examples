"""DS-MPNN 分布式封装：领域分解 + overlap 通信 + 梯度聚合。

论文 Sec 3 与 Appendix F Algorithm 1:
- 计算域按坐标划分为 nproc 个子域，每个 GPU 一个子域，重叠长度 l（默认 l=r）。
- 每 hop 在重叠区域通信 latent 节点属性（Comm(ib, Omega, v_L)）与输出值（Comm(ib, Omega, v)）。
- 损失对 interior points 求和（L = sum_i L_local），梯度跨 GPU 聚合后同步更新。

当前实现采用 torch.distributed（gloo 后端在 CPU 多进程可运行，rccl 在 DCU）：
- 每个 rank 拥有一个子域的图样本。
- 通过 node_index（全局索引）识别重叠节点，使用 all_gather 同步重叠节点的 latent 值。
- 梯度通过 DistributedDataParallel 语义或手动 all_reduce 聚合。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist


def get_global_comm_overlap(local_latent: torch.Tensor, node_index: torch.Tensor,
                            world_size: int, rank: int) -> torch.Tensor:
    """通过 all_gather 同步重叠区域的 latent 节点属性。

    简单实现：将各 rank 的 latent 与 node_index 全部收集，构建全局映射，
    然后每个 rank 用自己的 node_index 从全局 latent 中取回（含 overlap 的最近邻值）。
    这里为保持 shape 一致性，直接 all_gather latent 与 index，再按全局索引重建。
    """
    world_size = dist.get_world_size()
    gathered_latent = [torch.zeros_like(local_latent) for _ in range(world_size)]
    gathered_idx = [torch.zeros_like(node_index) for _ in range(world_size)]
    dist.all_gather(gathered_latent, local_latent)
    dist.all_gather(gathered_idx, node_index)
    # 构建全局索引 -> latent 映射
    all_lat = torch.cat(gathered_latent, dim=0)
    all_idx = torch.cat(gathered_idx, dim=0)
    # 用本地 node_index 查全局映射（确保重叠节点取到邻域值）
    # 注意：gather 后索引可能重复，这里取第一个匹配（简化实现，冒烟验证用）
    out = torch.zeros_like(local_latent)
    for i, gid in enumerate(node_index.tolist()):
        mask = (all_idx == gid)
        if mask.any():
            out[i] = all_lat[mask][0]
    return out


class DS_MPNN(nn.Module):
    """DS-MPNN：在 S-MPNN 基础上加入分布式通信（latent 与输出的 overlap 同步）。"""

    def __init__(self, base_model: nn.Module, use_communication: bool = True):
        super().__init__()
        self.base = base_model
        self.use_communication = use_communication

    def forward(self, data) -> torch.Tensor:
        """data: PyG Data（含 node_index 全局索引）。"""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_index = data.node_index if hasattr(data, 'node_index') else None

        latent = self.base.encoder(x)
        # 分布式通信：每 hop 后同步 latent 重叠区
        for k in range(self.base.hops):
            res = latent
            latent = self.base.kernel.conv(latent, edge_index, edge_attr) + res
            if self.use_communication and dist.is_initialized() and dist.get_world_size() > 1:
                latent = get_global_comm_overlap(latent, node_index,
                                                 dist.get_world_size(), dist.get_rank())
            src = latent[edge_index[0]]
            dst = latent[edge_index[1]]
            edge_attr = self.base.kernel.edge_update_proj(src - dst)
        out = self.base.decoder(latent)
        return out
