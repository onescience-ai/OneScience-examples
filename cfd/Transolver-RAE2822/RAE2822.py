import random
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import copy
from torch.utils.data import DistributedSampler
import logging
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager


class RAE2822Dataset(BaseDataset):
    """
    RAE2822 翼型数据集
    
    包含跨音速 RAE2822 翼型的 CFD 模拟结果，包括压力场和速度场。
    数据存储在规则网格上，支持随机采样和 Clenshaw-Curtis 采样两种数据集。
    返回 PyG Data 对象以适配 Transolver 模型。
    """
    
    DOMAIN = "cfd"
    TASK = "regression"
    DATA_FORMATS = ["npy"]

    def __init__(self, config: Union[Dict[str, Any]], mode: str = 'train', coef_norm: Optional[Tuple] = None):
        """
        初始化 RAE2822 数据集
        
        Parameters
        ----------
        config : Dict[str, Any]
            数据集配置
        mode : str, optional
            'train', 'val', 或 'test'
        coef_norm : tuple, optional
            (mean_in, std_in, mean_out, std_out) 归一化系数。
            如果为 'train' 且此项为 None，将尝试加载或计算。
            如果为 'val'/'test'，必须提供此项。
        """
        self.mode = mode
        self._provided_coef_norm = coef_norm
        self.data_list_names = []
        self.coef_norm = None
        self.dist = DistributedManager()
        
        super().__init__(config)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
                
        self._init_paths()
        self._load_data()
        self._load_metadata()
        if self.dist.rank == 0:
            self.logger.info(f"[{self.mode}] RAE2822 dataset initialized.")
            self.logger.info(f"[{self.mode}] Found {len(self.data_list_names)} samples.")

    def _init_paths(self):
        """
        初始化数据路径
        """
        super()._init_paths()
        
        self.db_path = self.data_path / self.config.source.db_name
        self.airfoil_path = self.data_path / "airfoil.npy"
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found at: {self.db_path}")
        if not self.airfoil_path.exists():
            raise FileNotFoundError(f"Airfoil file not found at: {self.airfoil_path}")

    def _load_data(self):
        """
        加载数据集并根据模式划分
        """
        if self.dist.rank == 0:
            self.logger.info(f"Loading raw data from {self.db_path}...")
        
        data_dict = np.load(self.db_path, allow_pickle=True).item()
        
        self.pressure = data_dict['Pressure']
        self.vx = data_dict['Vx']
        self.vy = data_dict['Vy']
        self.x_coord = data_dict['Xcoordinate']
        self.y_coord = data_dict['Ycoordinate']
        self.vinf = data_dict['Vinf']
        self.alpha = data_dict['Alpha']
        self.indices = data_dict['idx']
        
        n_samples = len(self.indices)
        
        split_ratio = self.config.data.split_ratio
        split_idx = int(n_samples * split_ratio)
        
        indices = list(range(n_samples))
        seed = self.config.data.seed
        random.Random(seed).shuffle(indices)
        
        if self.mode == 'train':
            selected_indices = indices[:split_idx]
        elif self.mode == 'val':
            selected_indices = indices[split_idx:split_idx + int(n_samples * (1 - split_ratio) / 2)]
        elif self.mode == 'test':
            selected_indices = indices[split_idx + int(n_samples * (1 - split_ratio) / 2):]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        self.data_list_names = selected_indices

    def _load_metadata(self):
        """
        加载或计算归一化系数（使用向量化计算，快速高效）
        """
        stats_dir = Path(self.config.source.stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        mean_in_path = stats_dir / "mean_in.npy"
        std_in_path = stats_dir / "std_in.npy"
        mean_out_path = stats_dir / "mean_out.npy"
        std_out_path = stats_dir / "std_out.npy"
        
        if self._provided_coef_norm:
            if self.dist.rank == 0:
                self.logger.debug(f"[{self.mode}] Using provided normalization coefficients.")
            self.coef_norm = self._provided_coef_norm
            return

        if mean_in_path.exists() and std_in_path.exists() and mean_out_path.exists() and std_out_path.exists():
            if self.dist.rank == 0:
                self.logger.info(f"[{self.mode}] Loading normalization stats from {stats_dir}")
            mean_in = np.load(mean_in_path)
            std_in = np.load(std_in_path)
            mean_out = np.load(mean_out_path)
            std_out = np.load(std_out_path)
            self.coef_norm = (mean_in, std_in, mean_out, std_out)
        elif self.mode == 'train':
            if self.dist.rank == 0:
                self.logger.warning(f"[{self.mode}] Stats not found. Calculating normalization stats (vectorized)...")
            self.coef_norm = self._calculate_normalization_vectorized()
            np.save(mean_in_path, self.coef_norm[0])
            np.save(std_in_path, self.coef_norm[1])
            np.save(mean_out_path, self.coef_norm[2])
            np.save(std_out_path, self.coef_norm[3])
            if self.dist.rank == 0:
                self.logger.info(f"[{self.mode}] Saved normalization stats to {stats_dir}")
        else:
            raise FileNotFoundError(f"[{self.mode}] Normalization stats not found in {stats_dir}, and mode is not 'train'.")

    def _calculate_normalization_vectorized(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        使用向量化计算归一化统计量（快速高效）
        一次性计算所有训练集样本的均值和标准差
        """
        if self.mode != 'train':
            raise RuntimeError("Normalization calculation should only be done on the training set.")
        
        # 获取训练集索引
        train_indices = self.data_list_names
        
        # 向量化提取所有训练集数据
        train_pressure = self.pressure[train_indices]
        train_vx = self.vx[train_indices]
        train_vy = self.vy[train_indices]
        train_x_coord = self.x_coord[train_indices]
        train_y_coord = self.y_coord[train_indices]
        train_vinf = self.vinf[train_indices]
        train_alpha = np.deg2rad(self.alpha[train_indices])
        
        # 构建输入特征：[pos_x, pos_y, vinf, alpha, nx, ny, boundary_mask]
        # vinf 和 alpha 需要广播到所有点
        n_train = len(train_indices)
        n_points = train_x_coord.shape[1]
        
        # 创建 vinf 和 alpha 的广播版本（2维）
        vinf_broadcast = train_vinf[:, np.newaxis]
        alpha_broadcast = train_alpha[:, np.newaxis]
        
        # 计算法向量（简单近似：使用坐标梯度）
        # 对于规则网格，法向量可以近似为 [0, 0, 1]
        nx = np.zeros_like(train_x_coord)
        ny = np.zeros_like(train_y_coord)
        
        # 边界mask：简单判断是否为边界点（这里简化处理，实际需要根据翼型几何判断）
        boundary_mask = np.zeros_like(train_x_coord)
        
        # 构建输入特征
        init = np.stack([
            train_x_coord,
            train_y_coord,
            np.broadcast_to(vinf_broadcast, (n_train, n_points)),
            np.broadcast_to(alpha_broadcast, (n_train, n_points)),
            nx,
            ny,
            boundary_mask
        ], axis=-1)
        
        # 构建目标特征：[pressure, vx, vy, nu]
        # 添加湍流粘度 nu（这里简化为 0，因为数据集中没有）
        nu = np.zeros_like(train_pressure)
        target = np.stack([
            train_pressure,
            train_vx,
            train_vy,
            nu
        ], axis=-1)
        
        # 向量化计算均值
        mean_in = init.mean(axis=(0, 1), dtype=np.double)
        mean_out = target.mean(axis=(0, 1), dtype=np.double)
        
        # 向量化计算标准差
        std_in = init.std(axis=(0, 1), dtype=np.double)
        std_out = target.std(axis=(0, 1), dtype=np.double)
        
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        std_in = std_in.astype(np.single)
        std_out = std_out.astype(np.single)
        
        return (mean_in, std_in, mean_out, std_out)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list_names)

    def __getitem__(self, idx: int) -> Data:
        """
        获取单个样本，返回 PyG Data 对象
        
        Returns
        -------
        Data
            PyG Data 对象，包含以下属性：
            - pos: 网格点坐标 [n_points, 2]
            - x: 输入特征 [n_points, 7] (pos_x, pos_y, vinf, alpha, nx, ny, boundary_mask)
            - y: 目标场 [n_points, 4] (pressure, vx, vy, nu)
            - surf: 表面点mask [n_points]
        """
        sample_idx = self.data_list_names[idx]
        
        # 每个样本有独立的坐标，使用 sample_idx 索引
        pos_x = self.x_coord[sample_idx]
        pos_y = self.y_coord[sample_idx]
        pos = np.stack([pos_x, pos_y], axis=-1)
        
        vinf = self.vinf[sample_idx]
        alpha = np.deg2rad(self.alpha[sample_idx])
        
        vinf_field = np.full_like(pos_x, vinf)
        alpha_field = np.full_like(pos_x, alpha)
        
        # 计算法向量（简化处理）
        nx = np.zeros_like(pos_x)
        ny = np.zeros_like(pos_y)
        
        # 边界mask（简化：假设所有点都是体点，没有表面点）
        # 实际应用中应该根据翼型几何判断哪些点是表面点
        boundary_mask = np.zeros_like(pos_x)
        surf = np.zeros_like(pos_x, dtype=bool)
        
        # 构建输入特征
        x = np.stack([pos_x, pos_y, vinf_field, alpha_field, nx, ny, boundary_mask], axis=-1)
        
        # 构建目标特征（添加湍流粘度 nu）
        nu = np.zeros_like(self.pressure[sample_idx])
        y = np.stack([
            self.pressure[sample_idx],
            self.vx[sample_idx],
            self.vy[sample_idx],
            nu
        ], axis=-1)
        
        # 归一化
        if self.coef_norm:
            mean_in, std_in, mean_out, std_out = self.coef_norm
            x = (x - mean_in) / (std_in + 1e-8)
            y = (y - mean_out) / (std_out + 1e-8)
        
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        pos = torch.tensor(pos, dtype=torch.float)
        surf = torch.tensor(surf, dtype=torch.bool)
        
        return Data(x=x, y=y, pos=pos, surf=surf)


class RAE2822Datapipe:
    """
    RAE2822 数据管道
    """
    
    def __init__(self, params: Dict[str, Any], distributed: bool = False):
        self.config = params
        self.distributed = distributed
        
        self.train_dataset = RAE2822Dataset(copy.deepcopy(params), mode='train')
        self.coef_norm = self.train_dataset.coef_norm
        self.val_dataset = RAE2822Dataset(copy.deepcopy(params), mode='val', coef_norm=self.coef_norm)
        self.test_dataset = RAE2822Dataset(copy.deepcopy(params), mode='test', coef_norm=self.coef_norm)

    def train_dataloader(self):
        """返回训练数据加载器"""
        sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True
        ), sampler

    def val_dataloader(self):
        """返回验证数据加载器"""
        sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.distributed else None
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True
        ), sampler

    def test_dataloader(self):
        """返回测试数据加载器"""
        sampler = DistributedSampler(self.test_dataset, shuffle=False) if self.distributed else None
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=True
        ), sampler