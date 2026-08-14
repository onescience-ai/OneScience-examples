---
datasets:
- OneScience/airfrans
language:
- en
- zh
license: Apache License 2.0
tags:
- OneScience
- message-passing-neural-network
- graph-neural-network
- CFD
tasks: []
---

<p align="center"><strong><span style="font-size: 30px;">DSMPNN-DarcyFlow</span></strong></p>

# 模型介绍

DS-MPNN（Sampling-based Distributed Training with Message Passing Neural Network）复现模型，来自 arXiv:2402.15106。
本模型实现边缘条件图卷积消息传递神经网络（S-MPNN），用于 2-D Darcy flow PDE 代理建模（映射扩散系数场 a → 解场 u）。

# 模型描述

主要模型文件：
- model/mpnn.py：S-MPNN 完整模型（Encoder + Edge-conditioned conv Kphi + Decoder）
- model/kernel.py：边缘条件图卷积 kernel（论文 Eq 1）
- model/encoder.py / decoder.py：编解码 MLP
- model/gcn.py：GCN baseline
- model/ds_mpnn.py：DS-MPNN 分布式封装（领域分解 + overlap 通信）

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 模型训练 | 使用 Darcy flow 数据训练 MPNN 代理模型 |
| 模型推理 | 加载权重，预测 Darcy flow 解场 |
| 模型评估 | 计算 RMSE/L1 指标对比 S-MPNN 与 GCN |

# 使用说明

## OneCode 使用

可接入 OneCode 智能体，通过 skill 完成训练、推理与评估。

## 手动安装使用

### 硬件要求

支持 GPU（NVIDIA CUDA）与 DCU（AMD Hygon）环境，也可在 CPU 上运行（速度较慢）。

### 下载模型包

```shell
modelscope download --model OneScience/DSMPNN-DarcyFlow
```

### 安装运行环境

```shell
conda create -n dsmpnn python=3.10 -y
conda activate dsmpnn
pip install torch torch_geometric numpy scipy pyyaml
```

### 训练数据介绍

本模型使用 AirfRANS 数据集（二维翼型不可压缩 RANS 亚音速仿真点云）。

```bash
modelscope download --dataset OneScience/airfrans --local_dir ./data
```

### 训练

```shell
python scripts/train.py --config conf/darcy_tier1.yaml --model smpnn
```

### 训练权重

- weight/smpnn_final.pt：S-MPNN 训练权重（677k 参数）
- weight/gcn_final.pt：GCN baseline 权重（575k 参数）

### 推理

```shell
python scripts/evaluate.py --config conf/darcy_tier1.yaml --checkpoint weight/smpnn_final.pt
```

### 评估和可视化

```shell
python scripts/test_smoke.py --config conf/darcy_smoke.yaml
python scripts/evaluate.py --config conf/darcy_tier1.yaml --checkpoint weight/gcn_final.pt
```

# OneScience 官方信息

| 资源 | 链接 |
| --- | --- |
| Gitee | https://gitee.com/onescience |
| GitHub | https://github.com/onescience-ai |

# 引用与许可证

本模型复现自：Sampling-based Distributed Training with Message Passing Neural Network, Priyesh Kakka et al., arXiv:2402.15106 (2024)。

