"""Darcy flow 数据生成器。

论文 App B: -nabla(a(x) nabla u(x)) = f(x), u|dOmega = 0 on unit box (0,1)^2, 映射 a -> u。
采用经典 GNO/FNO 风格：扩散系数场 a(x) 由随机过程生成（截断 Karhunen-Loeve 型随机场），
解场 u(x) 通过中心差分近似求解椭圆方程（统一使用有限差分，保证可复现）。
"""

from __future__ import annotations

import numpy as np


def generate_diffusion_field(grid_size: int, seed: int, modes: int = 16, sigma: float = 1.0) -> np.ndarray:
    """生成扩散系数场 a(x) = sigma * exp(w)，w 为随机 Fourier 系数场。"""
    rng = np.random.default_rng(seed)
    coords = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    w = np.zeros_like(X)
    for i in range(1, modes + 1):
        for j in range(1, modes + 1):
            amp = rng.standard_normal() / (i * j) ** 1.5
            phase = rng.uniform(0, 2 * np.pi)
            w += amp * np.sin(2 * np.pi * i * X + phase) * np.cos(2 * np.pi * j * Y)
    # 平移保证 a > 0
    a = sigma * np.exp(w - w.mean())
    return a


def solve_darcy(a: np.ndarray, f: np.ndarray | None = None) -> np.ndarray:
    """用中心差分求解 -nabla(a nabla u) = f, Dirichlet 零边界。

    返回解场 u（shape 与 a 相同）。f 默认取 1.0。
    """
    n = a.shape[0]
    h = 1.0 / (n - 1)
    if f is None:
        f = np.ones_like(a)

    # 系数平均（谐波）在交错网格节点
    a_i = 0.5 * (a[:-1, :] + a[1:, :])   # x 方向交错
    a_j = 0.5 * (a[:, :-1] + a[:, 1:])   # y 方向交错

    # 组装稀疏矩阵（中心差分五点格式）
    from scipy import sparse
    n_inner = (n - 2) ** 2
    idx = lambda i, j: (i - 1) * (n - 2) + (j - 1)
    rows, cols, vals, rhs = [], [], [], []

    def add(i, j, coef, r):
        rows.append(idx(i, j)); cols.append(idx(i, j)); vals.append(coef); rhs.append(r)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            # -d/dx(a du/dx) - d/dy(a du/dy) = f
            # a_{i+1/2}(u_{i+1}-u_i) - a_{i-1/2}(u_i - u_{i-1}) + a_{j+1/2}(u_{j+1}-u_j) - a_{j-1/2}(u_j-u_{j-1}) = f h^2
            apx = a_i[i, j]   # a at i+1/2, j
            amx = a_i[i - 1, j]  # a at i-1/2, j
            apy = a_j[i, j]   # a at i, j+1/2
            amy = a_j[i, j - 1]  # a at i, j-1/2

            diag = apx + amx + apy + amy
            add(i, j, diag, f[i, j] * h * h)
            if i + 1 < n - 1:
                rows.append(idx(i + 1, j)); cols.append(idx(i, j)); vals.append(-apx)
            if i - 1 >= 1:
                rows.append(idx(i - 1, j)); cols.append(idx(i, j)); vals.append(-amx)
            if j + 1 < n - 1:
                rows.append(idx(i, j + 1)); cols.append(idx(i, j)); vals.append(-apy)
            if j - 1 >= 1:
                rows.append(idx(i, j - 1)); cols.append(idx(i, j)); vals.append(-amy)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_inner, n_inner)).tocsr()
    b = np.array(rhs)
    u_inner = sparse.linalg.spsolve(A, b)
    u = np.zeros((n, n))
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            u[i, j] = u_inner[idx(i, j)]
    return u


class DarcyFlowGenerator:
    """生成 Darcy flow 数据集：每个样本 = (node_attr, target, coords)。

    node_attr 按论文为 R3 = (x, y, a)；target 为解场 u。
    """

    def __init__(self, grid_size: int = 421, seed: int = 42, modes: int = 16):
        self.grid_size = grid_size
        self.seed = seed
        self.modes = modes
        self.coords = np.linspace(0.0, 1.0, grid_size)

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成第 index 个样本：返回 (node_attr, target, grid_coords)。

        node_attr shape [N, 3]（x,y,a）；target shape [N]（u）；coords shape [N,2]。
        """
        seed = self.seed + index
        a = generate_diffusion_field(self.grid_size, seed, modes=self.modes)
        u = solve_darcy(a)
        X, Y = np.meshgrid(self.coords, self.coords, indexing="ij")
        # 展平所有网格节点（含边界，损失用 interior mask 处理）
        xf, yf, af, uf = X.ravel(), Y.ravel(), a.ravel(), u.ravel()
        node_attr = np.stack([xf, yf, af], axis=1)
        return node_attr, uf, np.stack([xf, yf], axis=1)


def build_darcy_dataset(train_samples: int, test_samples: int, grid_size: int,
                        seed: int = 42, modes: int = 16, nproc: int = 1):
    """构建完整 Darcy 数据集（训练 + 测试），支持多进程并行。"""
    gen = DarcyFlowGenerator(grid_size=grid_size, seed=seed, modes=modes)
    if nproc <= 1:
        train = [gen.sample(i) for i in range(train_samples)]
        test = [gen.sample(train_samples + i) for i in range(test_samples)]
        return train, test
    import multiprocessing as mp
    with mp.Pool(nproc) as pool:
        train = pool.map(gen.sample, range(train_samples))
        test = pool.map(gen.sample, range(train_samples, train_samples + test_samples))
    return train, test
