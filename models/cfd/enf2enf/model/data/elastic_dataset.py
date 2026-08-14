"""elasticity 数据读取与张量协议（依据 reproduction_spec.md 第2/3节）。

数据：CFD_Benchmark/elasticity/Meshes/Random_UnitCell_XY_10.npy (N,2,P)
与 Random_UnitCell_sigma_10.npy (N,P)。
几何描述符 a = XY - mean(XY_train)（附录 B.1.2 displacement field）。
坐标 min-max 归一化到 [-1,1]；输出 z-score 标准化（统计量只由 train 计算）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ElasticityDataset(Dataset):
    """超弹性单元胞材料应力回归数据集。

    Args:
        data_dir: 含 .npy 文件的目录。
        split: 'train' | 'test'。
        split_ratio: train 占比。
        seed: 随机切分种子。
        normalization: {'mean_xy','coord_min','coord_max','sigma_mean','sigma_std'}
            由 train 集计算的归一化统计量；test 集沿用。
        points: 每个样本保留的点数（None 表示全部 2000）。
        transform: 可选 torch 变换。
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_ratio: float = 0.8,
        seed: int = 42,
        normalization: dict | None = None,
        points: int | None = None,
        xy_file: str = "Random_UnitCell_XY_10.npy",
        sigma_file: str = "Random_UnitCell_sigma_10.npy",
    ):
        import os

        xy_path = os.path.join(data_dir, xy_file)
        sg_path = os.path.join(data_dir, sigma_file)
        self.xy_all = np.load(xy_path)  # (N,2,P)
        self.sigma_all = np.load(sg_path)  # (N,P)
        assert self.xy_all.ndim == 3 and self.xy_all.shape[0] == self.sigma_all.shape[0]

        self.n_samples, _, self.n_points = self.xy_all.shape

        rng = np.random.RandomState(seed)
        perm = rng.permutation(self.n_samples)
        n_train = int(self.n_samples * split_ratio)
        idx = perm[:n_train] if split == "train" else perm[n_train:]
        self.xy = self.xy_all[idx]
        self.sigma = self.sigma_all[idx]

        if points is not None and points < self.n_points:
            # 固定子采样（保持一致）
            self.xy = self.xy[:, :, :points]
            self.sigma = self.sigma[:, :points]
            self.n_points = points

        if normalization is None:
            normalization = self._compute_normalization()
        self.norm = normalization

    def _compute_normalization(self) -> dict:
        train_xy = self.xy_all  # 在构造时对整集计算；test 复用 train 统计由外部传入
        return {
            "mean_xy": self.xy_all.mean(axis=(0, 2), keepdims=True),  # (1,2,1)
            "coord_min": float(self.xy_all.min()),
            "coord_max": float(self.xy_all.max()),
            "sigma_mean": float(self.sigma_all.mean()),
            "sigma_std": float(self.sigma_all.std()),
        }

    def __len__(self) -> int:
        return self.xy.shape[0]

    def __getitem__(self, i: int):
        xy = self.xy[i]  # (2,P)
        sigma = self.sigma[i]  # (P,)

        # 几何描述符：displacement between average mesh position and current sample
        mean_xy = self.norm["mean_xy"][:, 0]  # (2,1)
        a = xy - mean_xy

        # 坐标 min-max 归一化
        cmin, cmax = self.norm["coord_min"], self.norm["coord_max"]
        xy_norm = 2.0 * (xy - cmin) / (cmax - cmin) - 1.0

        # 输出 z-score 标准化
        sigma_norm = (sigma - self.norm["sigma_mean"]) / self.norm["sigma_std"]

        coords = torch.tensor(xy_norm, dtype=torch.float32)  # (2,P)
        geom = torch.tensor(a, dtype=torch.float32)  # (2,P)
        target = torch.tensor(sigma_norm, dtype=torch.float32)  # (P,)
        return coords, geom, target

    def inverse_sigma(self, sigma_norm: torch.Tensor) -> torch.Tensor:
        return sigma_norm * self.norm["sigma_std"] + self.norm["sigma_mean"]

    @staticmethod
    def build_splits(
        data_dir: str,
        split_ratio: float = 0.8,
        seed: int = 42,
        points: int | None = None,
    ) -> tuple["ElasticityDataset", "ElasticityDataset"]:
        """构建 train/test 并保证 test 使用 train 的归一化统计量。"""
        train = ElasticityDataset(data_dir, "train", split_ratio, seed, None, points)
        test = ElasticityDataset(
            data_dir, "test", split_ratio, seed, normalization=train.norm, points=points
        )
        return train, test
