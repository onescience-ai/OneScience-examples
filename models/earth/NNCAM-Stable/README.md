<p align="center"><strong><span style="font-size: 30px;">NNCAM-Stable</span></strong></p>

# 模型介绍

NNCAM-Stable 使用神经网络替代 GCM 中的大气湿物理与辐射过程。模型面向多年稳定在线气候模拟。

论文：Stable climate simulations using a realistic general circulation model with neural network parameterizations for atmospheric moist physics and radiation processes  
https://doi.org/10.5194/gmd-15-3923-2022

# 模型描述

该模型由清华大学和 Scripps Institution of Oceanography 的研究团队提出。模型使用两年 SPCAM 的 122 维大气柱输入和 68 维倾向与辐射输出训练。模型适用于 CAM5.2 湿物理与辐射在线参数化以及多年稳定气候模拟。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 湿物理参数化 | 预测 30 层水汽和干静能倾向。 |
| 辐射仿真 | 预测地表和大气顶辐射通量。 |
| 在线稳定性 | 验证神经参数化长期耦合稳定性。 |
| ModelScope/OneCode 运行 | 验证结构化数据、训练、推理、气候指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/NNCAM-Stable --local_dir ./NNCAM-Stable
cd NNCAM-Stable
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

论文使用真实地理配置的 SPCAM 模拟数据，每个样本包含 122 维大气柱输入和 68 维湿物理倾向与辐射输出。虚拟数据保持全部输入输出维度、30 个垂直层和三个独立参数化网络，仅减少样本数量、隐藏宽度和训练轮数。该数据仅用于验证训练、推理和评估流程，不代表 SPCAM 官方数据分布或论文性能。

```bash
python scripts/fake_data.py
```

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py
```

训练分别优化水汽倾向、干静能倾向和辐射通量网络，单卡和双进程 DDP 均已验证通过。训练生成单一可恢复 checkpoint，并记录整体 MSE。训练结果保存到：

```text
result/checkpoints/nncam_stable.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。作者代码归档位于 https://doi.org/10.5281/zenodo.5596273 ，训练数据位于 https://doi.org/10.5281/zenodo.5625616 ；论文未单独提供可直接加载的官方预训练权重文件。

### 推理

```bash
python scripts/inference.py
```

推理恢复 checkpoint，并输出 30 层水汽倾向、30 层干静能倾向和 8 个辐射通量。输出 shape 为 `[128,68]`，且已通过有限数值检查。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算全部倾向与辐射输出的整体 RMSE，并生成输出剖面对比图。指标和 PNG 均通过有效性检查。评估结果保存到：

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 NNCAM-Stable 公开规格的独立工程复现版本。

SPCAM v2、NNCAM 源代码、模型产物以及训练与测试数据应遵循对应归档资源的许可证及使用条款。
