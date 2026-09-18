<p align="center">
  <strong>
    <span style="font-size: 30px;">PGT</span>
  </strong>
</p>

# 模型介绍

PGT（Physics-Guided Transformer）是面向 PINN（物理信息神经网络）的稀疏物理场重建模型。该模型将热核格林函数（heat-kernel Green's function）导出的加性偏置直接嵌入自注意力 logits，编码扩散动力学与时序因果性；查询坐标通过 cross-attention 关注物理条件化 context token，并由 FiLM 调制的正弦隐式网络解码输出连续场。

论文：Physics-Guided Transformer (PGT): Physics-Aware Attention Mechanism for PINNs
https://arxiv.org/abs/2603.27929

# 模型描述

PGT 采用 Transformer 架构，核心组件如下：

- `pgt.py`：PGT 主模型（物理引导 Transformer 编码器 + cross-attention 查询条件化 + FiLM-SIREN 解码器）
- `physics_bias.py`：热核导出的物理偏置 Gamma（Eq.7-8）与因果掩码
- `film_siren.py`：FiLM 调制的正弦隐式网络（SIREN）解码器（Eq.11-14）
- `losses.py`：不确定性加权复合损失（Eq.15-18）与 PDE 算子（1D 热方程、2D NS 动量+连续性）
- `data/heat1d.py`、`data/cylinder_wake.py`：1D 热方程解析数据生成与 2D NS 圆柱尾迹数据加载

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 稀疏物理场重建训练 | 使用 1D 热方程观测点或 2D NS 圆柱尾迹 1500 采样点训练 PGT |
| 模型推理 | 加载权重对任意查询坐标输出连续场值 |
| 评估与可视化 | 计算 Rel-L2（总/分项）与 PDE residual |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/PGT --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem、bio、all（全领域）
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem（全领域）
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

- 1D 热方程：由解析解 `u(x,t)=exp(-nu*(n*pi)^2*t)*sin(n*pi*x)` 生成，x,t 在 [0,1] 内随机采样 M 个观测点。
- 2D NS 圆柱尾迹：使用 cylinder_nektar_wake.mat（5000 空间点 × 200 时间步的 u/v/p 场），随机采样 1500 个时空点。

### 训练

```bash
# 1D 热方程稀疏重建（M=100 观测）
python scripts/train.py --config conf/pgt.yaml --task heat1d --device dcu --epochs 500

# 2D NS 圆柱尾迹重建（1500 采样）
python scripts/train.py --config conf/pgt.yaml --task cylinder --device dcu --epochs 100
```

训练会在 `outputs/pgt_tier1/checkpoints/` 下保存 `pgt_heat1d.pt` 与 `pgt_cylinder.pt`。

### 训练权重

- `weight/pgt_heat1d.pt`：1D 热方程模型权重
- `weight/pgt_cylinder.pt`：2D NS 圆柱尾迹模型权重

### 推理

```bash
# 对任意查询坐标（N, coord_dim 的 .npy 文件）输出场值
python scripts/infer.py --config conf/pgt.yaml --checkpoint weight/pgt_heat1d.pt --task heat1d     --coords-file coords.npy --output pred.npy
```

推理结果保存至指定的 .npy 文件。

### 评估和可视化

```bash
# 1D 热方程评估（Rel-L2、PDE residual）
python scripts/evaluate.py --config conf/pgt.yaml --checkpoint weight/pgt_heat1d.pt --task heat1d --device dcu

# 2D NS 圆柱尾迹评估
python scripts/evaluate.py --config conf/pgt.yaml --checkpoint weight/pgt_cylinder.pt --task cylinder --device dcu
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为以下论文的复现版本：

- Physics-Guided Transformer (PGT): Physics-Aware Attention Mechanism for PINNs
  Ehsan Zeraatkar, Rodion Podorozhny, Jelena Tešić (Texas State University)
  arXiv:2603.27929, https://arxiv.org/abs/2603.27929

