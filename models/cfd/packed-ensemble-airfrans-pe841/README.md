---
datasets:
- OneScience/airfrans
language:
- en
- zh
license: Apache License 2.0
tags:
- OneScience
- Packed-Ensemble
- AirfRANS
- CFD
- surrogate-model
tasks: []
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">packed-ensemble-airfrans-pe841</span>
  </strong>
</p>

# 模型介绍

本模型是论文 "Packed-Ensemble Surrogate Models for Fluid Flow Estimation Around Airfoil Geometries"（arXiv:2312.13403）的复现实现。论文研究 Packed-Ensembles（PE）代理模型用于加速翼型气动流场估算，通过嵌入多个并行小子网络来泛化 Deep Ensembles，在保持平滑性与不确定性的同时显著降低训练成本。论文中 **PE(8,4,1)** 被识别为该任务的最优配置，在超越其 Deep Ensemble 对应物的同时，训练时间加快了 25%。

本仓库复现 PE(8,4,1)：M=8, α=4, γ=1，架构 `(64,64,8,64,64,64,8,64,64)`，在 AirfRANS 数据集上进行点级（point-wise）流场回归（7 维输入 → 4 维输出）。

论文：https://arxiv.org/abs/2312.13403

# 模型描述

模型为 Packed-Ensemble 多层感知机（PE-MLP），使用分组（Packed）线性层实现多个并行子网络：

- 输入特征（7 维）：`pos_x, pos_y, u_inf_x, u_inf_y, dist, normal_x, normal_y`
- 输出目标（4 维）：`v_x, v_y, p, nut`
- 架构 `layers = (64,64,8,64,64,64,8,64,64)`，层间 ReLU 激活
- Packed-Ensemble 超参数：`num_estimators M=8`（并行子网络数）、`alpha=4.0`（容量调制）、`gamma=1`（稀疏调制）
- 损失：MSE；优化器：Adam（lr=0.01, weight_decay=1e-5）
- 推理时对 M 个子网络输出取平均，得到平滑的集成预测

主要模型文件：
- `model/model.py`：包含 `PackedLinear`、`PackedEnsembleMLP`、`BaselineMLP` 与 `model_factory`

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 流场回归训练 | 使用 AirfRANS 数据训练 Packed-Ensemble 代理模型 |
| 模型推理 | 加载权重对翼型网格点云预测速度/压力/湍流黏度 |
| 物理指标评估 | 计算 MSE、mean relative drag/lift 与 Spearman 相关 |
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
modelscope download --model OneScience/packed-ensemble-airfrans-pe841 --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai 
```

**GPU环境**
```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本模型使用 AirfRANS 数据集（二维翼型不可压缩 RANS 亚音速仿真点云）。

```bash
modelscope download --dataset OneScience/airfrans --local_dir ./data
```

### 训练

单卡：

```bash
python scripts/train.py \
  --model pe_mlp \
  --layers "64,64,8,64,64,64,8,64,64" \
  --num-estimators 8 --alpha 4 --gamma 1 \
  --lr 1e-2 --wd 1e-5 \
  --data-dir /path/to/airfrans/Dataset \
  --out-dir output/pe841
```

### 训练权重

- `weight/model_final.pt`：PE(8,4,1) 最终训练权重（约 9.2 MB）

### 推理

```bash
python scripts/evaluate.py \
  --checkpoint weight/model_final.pt \
  --data-dir /path/to/airfrans/Dataset \
  --split test --out output/eval.json
```

### 评估和可视化

```bash
python scripts/evaluate.py --checkpoint weight/model_final.pt --data-dir /path/to/airfrans/Dataset --split test --dcu
```

评估输出 MSE（x/y velocity、pressure、turbulent viscosity）、mean relative drag/lift 与 Spearman 相关，详见 `conf/eval_test.json`、`conf/eval_test_ood.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 Packed-Ensemble Surrogate Models for Fluid Flow Estimation Around Airfoil Geometries（arXiv:2312.13403）的复现版本。

