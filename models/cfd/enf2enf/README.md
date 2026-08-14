---
datasets:
- CFD_Benchmark/elasticity
language:
- en
- zh
license: Apache License 2.0
tags:
- OneScience
- PDE
- neural-field
- equivariant-neural-field
- operator-learning
- CFD
- hyperelastic
tasks: []
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">enf2enf</span>
  </strong>
</p>

# 模型介绍

enf2enf 是 arXiv:2504.18591《Geometry Aware Inference of Steady State PDEs Using Equivariant Neural Field Representations》的复现模型。该模型使用等变神经场（Equivariant Neural Fields, ENF）作为几何编码器，通过 CAVIA 元学习内循环把几何（如 signed distance field / 位移描述符）编码为 spatially anchored 的潜在点云，再结合全局工况参数，经 latent self-attention 与等变交叉注意力解码为连续物理场。实验面向稳态 PDE 代理（气动 AirfRANS 压力场与 hyper-elastic 材料应力场），具备 discretization-invariant 特性，可在任意查询坐标上连续解码。

论文：Geometry Aware Inference of Steady State PDEs Using Equivariant Neural Field Representations  
https://arxiv.org/abs/2504.18591

# 模型描述

enf2enf 基于 等变神经场（ENF）架构，采用 encoder-decoder 结构：

- **RFF 坐标编码**：`gamma(x)=[cos(Wx), sin(Wx)]`，encoder RFF(d=128, sigma=2)，decoder RFF(d=256, sigma=10)。
- **Encoder f_theta_a**：等变交叉注意力（Eq.4-6），Gaussian window sigma=0.1，heads=2，width=128；CAVIA 元学习内循环 K=3 步优化 latent features c_j（n_lat=9, l_dim=8）。
- **Latent Self-Attention**：2 blocks，residual + LayerNorm，width=256。
- **Decoder f_theta_u**：等变交叉注意力 + 条件 latent，输出连续物理场。
- **损失**：L^a（输入几何重建 MSE）+ L^u（物理场重建 MSE），分阶段训练。

本包内置 hyper-elastic 单元胞材料应力场回归权重（Tier 1 小规模训练）。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用 elasticity 单元胞数据训练 enf2enf（分阶段 encoder + decoder） |
| 模型推理 | 加载权重，在任意查询坐标解码物理场 |
| 模型评估 | 计算 Mean L2 Relative Error |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |

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
modelscope download --model OneScience/enf2enf --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai 
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

Hyper-elastic 单元胞材料数据（Geo-FNO 基准）：单位域 `[0,1]^2` 含中心任意形状 void，约 1000 点表示几何，输出为应力场；论文 1000 train + 200 test。本包使用 `CFD_Benchmark/elasticity/Meshes/` 数据（972 样本 × 2000 点），几何描述符为位移场 `XY - mean(XY_train)`，坐标 min-max 归一化到 [-1,1]，输出 z-score 标准化。请在 `conf/elasticity.yaml` 中把 `data.data_dir` 指向本地数据目录。

### 训练

运行时需让脚本找到 `model/` 下的包（将 model 目录加入 PYTHONPATH 或从包根执行）：

单卡（DCU/GPU）：

```bash
# 方法一：设置 PYTHONPATH
PYTHONPATH=./model python scripts/train.py --config conf/elasticity.yaml --phase all --epochs-encoder 20 --epochs-decoder 30 --max-samples 200
```

训练会在 `runs/elasticity/` 下保存 `encoder.pth` 与 `decoder.pth`，日志写入 `runs/elasticity/train.log`。

### 训练权重

- `weight/encoder.pth`（4.2M，encoder 权重）
- `weight/decoder.pth`（4.2M，decoder 权重）
- 训练规模：Tier 1（200 样本，encoder 20 epochs + decoder 30 epochs），测试集 Mean L2 Relative Error ≈ 0.3898（论文全量训练参考 0.0188，需更大规模训练对齐）。

### 推理

```bash
PYTHONPATH=./model python scripts/infer.py --config conf/elasticity.yaml --split test
```

推理结果保存至 `runs/elasticity/predictions/`（predictions.npy / targets.npy，原始单位）。

### 评估和可视化

```bash
PYTHONPATH=./model python scripts/evaluate.py --config conf/elasticity.yaml --split test
```

评估结果保存至 `runs/elasticity/metrics.json`（Mean L2 Relative Error）。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为 arXiv:2504.18591《Geometry Aware Inference of Steady State PDEs Using Equivariant Neural Field Representations》的复现版本。
- 论文原文采用 CC BY 4.0 许可。

