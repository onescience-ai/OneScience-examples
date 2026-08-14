---
datasets:
- OneScience/airfrans
language:
- en
- zh
license: Apache License 2.0
tags:
- OneScience
- aerofoil
- cfd
- point-cloud
- graph-neural-network
tasks: []
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">aerofoil-ml-bench</span>
  </strong>
</p>

# 模型介绍

aerofoil-ml-bench 复现论文 "Benchmarking machine learning models for predicting aerofoil performance"（arXiv:2504.15993, EWTEC 2025）。该工作比较了四种神经网络（MLP、PointNet、GraphSAGE、GUNet）用于预测翼型周围流场（密度、动量、能量、涡量）并进一步计算升力系数 CL 的性能。

论文：Benchmarking machine learning models for predicting aerofoil performance
https://arxiv.org/abs/2504.15993

注意：本仓库使用本地可用的 CFD_Benchmark/NACA_Cylinder 数据做端到端演示，并非论文原始 windAI_bench 数据集，结果与论文 Table 1-4 不可直接比较。

# 模型描述

包含论文中的四种模型结构，均由 encoder(MLP) + 主干 + decoder(MLP) 组成，输入为每节点的 6 维特征，输出为每节点的 5 维流场变量：

- `MLP.py` (models/NN.py)：多层感知机基线
- `PointNet.py`：点云 MLP + 全局最大池化基线
- `GraphSAGE.py`：图邻域聚合网络
- `GUNet.py`：多尺度图 U-Net

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 流场预测训练 | 使用翼型 CFD 数据训练上述四类模型 |
| 本地快速验证 | 检查数据读取、模型训练与推理 |
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
modelscope download --model OneScience/aerofoil-ml-bench --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本模型使用 AirfRANS 数据集（二维翼型不可压缩 RANS 亚音速仿真点云）。

```bash
modelscope download --dataset OneScience/airfrans --local_dir ./data
```

### 训练

```bash
python scripts/main.py MLP --data-dir <DATA_DIR> --out-dir metrics --foils 10 --epochs 30 --graph
```

训练权重保存在 `metrics/{foils}_samples/{model}/model.pt`。

### 训练权重

- `weight/MLP.pt`
- `weight/PointNet.pt`
- `weight/GraphSAGE.pt`
- `weight/GUNet.pt`

### 推理

```bash
python scripts/validation.py MLP --data-dir <DATA_DIR> --out-dir metrics --graph
```

推理输出测试集流场 RMSE（总/表面/流体/逐变量）及每样本推理耗时。

### 评估和可视化

```bash
python scripts/validation.py MLP --data-dir <DATA_DIR> --out-dir metrics --graph
```

（评估脚本 `validation.py` 计算测试 RMSE；面板法 CL 计算见 `scripts/post_proc/panel_method.py`，需要数据集包含翼型表面节点。）

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为 "Benchmarking machine learning models for predicting aerofoil performance"（arXiv:2504.15993）论文的复现版本。
- 官方代码：https://github.com/OllieS-PhD/Benchmark_Aerofoils

