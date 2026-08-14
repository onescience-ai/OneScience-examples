"""图构造器：Nyström 采样 + radius 图 + 边属性。

论文 Sec 2.1: 从完整域随机采样 s 个节点，以每个采样节点为中心构造 radius 图，
边数超过 ne 时随机采样 ne 条边；边属性 el = vi - vj。
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def sample_nodes(node_attr: torch.Tensor, s: int, rng: np.random.Generator) -> tuple[torch.Tensor, np.ndarray]:
    """从节点集中随机采样 s 个节点（Nyström 采样）。返回 (采样节点索引 tensor, 索引数组)。"""
    n = node_attr.shape[0]
    idx = rng.choice(n, size=min(s, n), replace=False)
    idx = np.sort(idx)
    return torch.from_numpy(idx).long(), idx


def build_radius_edge_index(coords: torch.Tensor, node_idx: np.ndarray, radius: float,
                            ne: int, rng: np.random.Generator) -> torch.Tensor:
    """对给定采样节点，构造以半径 radius 为邻域的边（仅在采样节点之间）。

    返回 edge_index shape [2, E]（连接中心采样节点->邻居采样节点，索引为采样集合内局部索引）。
    """
    c = coords.numpy()
    sampled_coords = c[node_idx]
    n_sampled = len(node_idx)
    rows, cols = [], []
    for k in range(n_sampled):
        # 计算中心到所有采样节点的距离
        d = np.linalg.norm(sampled_coords - sampled_coords[k][None, :], axis=1)
        neigh = np.where((d <= radius) & (d > 1e-9))[0]
        if len(neigh) > ne:
            neigh = rng.choice(neigh, size=ne, replace=False)
        for j in neigh:
            rows.append(k)
            cols.append(int(j))
    if len(rows) == 0:
        # 兜底：至少保留自环，避免空图
        rows = list(range(n_sampled))
        cols = list(range(n_sampled))
    edge_index = np.stack([rows, cols], axis=0)
    return torch.from_numpy(edge_index).long()


def compute_edge_attr(node_attr_sampled: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """边属性 el = vi - vj（节点属性差，含坐标与物理属性差）。"""
    src = node_attr_sampled[edge_index[0]]
    dst = node_attr_sampled[edge_index[1]]
    return src - dst


def build_graph(node_attr_full: torch.Tensor, target_full: torch.Tensor | None,
                coords_full: torch.Tensor, s: int, radius: float, ne: int,
                seed: int = 0) -> Data:
    """从完整域节点构造采样图。

    Args:
        node_attr_full: [N, C_in] 完整节点属性
        target_full: [N] 或 [N, C_out] 完整目标，可为 None（仅推理）
        coords_full: [N, 2] 坐标（用于半径构图）
        s: 采样节点数
        radius: kernel radius
        ne: 最大边数
        seed: 随机种子

    Returns:
        Data 对象：node_attr (sampled), edge_index, edge_attr, target, interior_mask, node_index
    """
    rng = np.random.default_rng(seed)
    sampled_idx, node_idx_arr = sample_nodes(node_attr_full, s, rng)
    edge_index = build_radius_edge_index(coords_full, node_idx_arr, radius, ne, rng)
    node_attr_s = node_attr_full[node_idx_arr]
    edge_attr = compute_edge_attr(node_attr_s, edge_index)

    data = Data(
        x=node_attr_s,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_index=torch.from_numpy(node_idx_arr).long(),
    )
    if target_full is not None:
        t = target_full[node_idx_arr]
        if t.ndim == 1:
            t = t[:, None]
        data.y = t
    # interior mask: 采样节点默认全部为 interior（loss 计入）
    data.interior_mask = torch.ones(node_attr_s.shape[0], dtype=torch.bool)
    return data


def build_graphs(node_attrs: list[torch.Tensor], targets: list[torch.Tensor],
                 coords_list: list[torch.Tensor], s: int, radius: float, ne: int,
                 seeds: list[int]) -> list[Data]:
    """批量构图。"""
    return [build_graph(na, t, c, s, radius, ne, seed)
            for na, t, c, seed in zip(node_attrs, targets, coords_list, seeds)]
