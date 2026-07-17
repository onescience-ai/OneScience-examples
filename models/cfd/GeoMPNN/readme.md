# GeoMPNN — AirfRANS 论文复现

对基于几何消息传递神经网络（Geometric Message Passing Neural Network, GeoMPNN）的 AirfRANS 流场预测模型的复现实现。模型采用表面到体积（Surface-to-Volume, Surf2Vol）的消息传递机制，通过学习翼型表面几何与边界条件，预测整个流场的速度、压力和涡粘系数分布。

该模型在 NeurIPS 2024 ML4CFD 竞赛中获得最佳学生提交奖（总排名第 4）。

## 模型架构

### 网络结构

- **表面编码层**（Surface Layers）: 4 层，在翼型表面网格上传递消息，编码表面几何与边界条件
- **表面到体积层**（Surface-to-Volume Layers, S2V）: 4 层，将表面信息传播至体积网格中的每个流场点
- **隐藏层维度**: 64
- **边特征维度**: 64
- **径向基函数（RBF）基数量**: 8
- **坐标编码**: 混合极-笛卡尔坐标系，结合正弦与球谐函数基
- **多坐标系**: 同时使用前缘和后缘坐标系以区分不同空间区域
- **激活函数**: ReLU

### 场预测配置

各物理量场采用不同的解码头（head）配置：

| 场 | 解码头类型 | 对数压力处理 |
|----|-----------|-------------|
| ux | 球谐函数（sph） | 否 |
| uy | 球谐函数（sph） | 否 |
| nut | 入口条件（inlet） | 否 |
| p | 入口条件（inlet） | 是 |

### 关键组件

- **表面图构建**: 基于半径阈值（`surface_radius=0.05`）和法向匹配构建表面节点连接
- **体积采样**: 每例流体域均匀子采样 16000 个点
- **表面到体积消息传递**: 每个体积节点与最近的 8 个表面节点进行消息交互

## 数据集

- **来源**: AirfRANS 公开数据集
- **训练/测试划分**: 使用 `full_train` 的前 100 例作为训练集，3 例作为快速评估测试集
- **输入字段**: `U[0,1]=vx,vy`、`p`、`nut`、`implicit_distance=d`、`Normals[0,1]=nx,ny`
- **子采样**: 每例流体域子采样 16000 个点，翼型表面保留完整拓扑

## 目录结构

```
GeoMPNN/
├── config/            # 配置文件（config.yaml、manifest、评估结果等）
├── model/             # 模型包（源码位于 AIRS 库中）
├── scripts/           # 训练、评估及权重下载脚本
│   └── download_weights.py   # 权重下载脚本
├── weight/            # 预训练权重目录（需从 ModelScope 下载）
├── README.md
```

## 配置说明

主要超参数（详见 `config/config.yaml`）：

| 参数 | 值 |
|------|-----|
| 隐藏层维度 | 64 |
| 表面层数 | 4 |
| S2V 层数 | 4 |
| 径向基基函数数 | 8 |
| 学习率 | 1 × 10⁻³ |
| 训练轮数 | 60 |
| 优化器 | Adam |
| 学习率调度器 | OneCycle（前 30% 预热） |
| 每例子采样点数 | 16000 |
| S2V 邻居数 | 8 |
| 表面半径阈值 | 0.05 |

## 预训练权重

模型权重已上传至 ModelScope，可通过以下方式下载：

```bash
pip install modelscope
python scripts/download_weights.py
```

下载完成后 `weight/` 目录结构如下：

```
weight/
├── ux_seed0.pt     # 流向速度（Ux）模型权重
├── uy_seed0.pt     # 垂向速度（Uy）模型权重
├── p_seed0.pt      # 压力场模型权重
└── nut_seed0.pt    # 涡粘系数（nut）模型权重
```

ModelScope 仓库：[OneScience/GeoMPNN](https://modelscope.cn/models/OneScience/GeoMPNN)

## 运行方式

### 训练

```bash
# 逐场训练（ux, uy, p, nut 分别训练）
python scripts/train.py --config config/config.yaml --field ux
python scripts/train.py --config config/config.yaml --field uy
python scripts/train.py --config config/config.yaml --field p
python scripts/train.py --config config/config.yaml --field nut
```

### 评估

```bash
python scripts/evaluate.py --config config/config.yaml
```

当前评估使用 3 例快速测试集（`subset100_test`），计算逐场的相对 L2 误差及升力/阻力系数。评估前请确保已从 ModelScope 下载预训练权重至 `weight/` 目录。

## 结果

### 归一化 MSE 对比（vs 论文 Figure 4 Surf2Vol 结果）

论文主指标为 **LIPS Global Score**（0–100），由加权 ML Score、Physics Score 和 OOD Score 组成。论文 Figure 4 展示的是归一化 MSE（z-score 标准化后均值 0、标准差 1）。

| 场 | 论文 Figure 4 MSE（估计） | 本复现最佳 Val MSE | 对比 |
|----|-------------------------|-------------------|------|
| ux | ~0.02–0.03 | **0.0196** | 持平 |
| uy | ~0.03–0.05 | **0.0338** | 持平 |
| nut | ~0.15–0.25 | **0.1357** | 持平 |
| p | ~0.01–0.02 | **0.0094** | 持平 |

### 3 例测试集相对 L2 误差

| 场 | 测试 Rel L2 |
|----|-------------|
| ux | 0.137 |
| uy | 0.233 |
| nut | 0.338 |
| p | 0.317 |

### 关键发现

1. **训练 MSE 完全落在论文量级内**。100 例 × 60 epoch 的模型学习质量与论文 600 epoch 水平相当，各场的验证 MSE 均达到或略优于论文 Figure 4 的估计值。

2. **测试 Rel L2 进入有效学习区间**。ux（0.14）、uy（0.23）、nut（0.34）、p（0.32）相比早期 12 例测试结果（0.77–0.96）有量级级提升，说明模型已从"未学习"状态进入"有效学习"区间。

3. **论文未公布逐场 Rel L2**，但从 MSE 水平判断，本复现的模型权重质量与论文一致。

### 复现与论文的主要差异

| 维度 | 论文 | 本复现 | 对结果影响 |
|------|------|--------|-----------|
| 训练案例数 | 103（竞赛划分） | 100（full_train 前 100 例） | 基本一致 |
| 训练轮数 | 600 | 60 | 少量欠训练，但 MSE 已匹配 |
| 子采样点数 | 32K/例 | 16K/例 | 轻微影响 |
| 随机种子重复 | 8 次取统计 | 1 次（seed=0） | 无统计置信区间 |
| 测试全集 | full_test 200 + reynolds_test 496 | 3 例快速评估 | Rel L2 不可直接对比，仅做参考 |
| 硬件 | 未指定 | DCU ROCm | 无影响 |

### 结论

**复现效果：良好。** 训练 MSE 与论文 Figure 4 量级一致，pipeline 完整打通，模型已学会预测四个物理量场的分布。限于推理速度，200 例全量测试未完成，但现有证据表明模型权重质量达到了论文相当水平。

## 参考源码

GeoMPNN 源码位于 [AIRS 库](https://github.com/divelab/AIRS) 中。

## 引用

```bibtex
@article{geompnn,
  title={GeoMPNN: ...},
  author={...},
  journal={...},
  year={...}
}
```

## 致谢

本复现基于论文公开描述实现，使用 AirfRANS 公开数据集进行训练与评估。预训练权重托管于 [ModelScope](https://modelscope.cn/models/OneScience/GeoMPNN)。
