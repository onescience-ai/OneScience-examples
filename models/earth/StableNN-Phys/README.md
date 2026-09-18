<p align="center">
  <strong><span style="font-size: 30px;">StableNN-Phys</span></strong>
</p>

# 模型介绍

StableNN-Phys 是面向大气单柱模式的神经网络统一物理参数化工程复现，可根据热力学柱状态和表面通量，以 3 小时时间间隔连续预测温湿状态演变。

论文：Prognostic Validation of a Neural Network Unified Physics Parameterization  
https://doi.org/10.1029/2018GL078510

# 模型描述

StableNN-Phys 对应的方法由华盛顿大学大气科学系的研究团队提出。模型接收 34 层液态水静能与总水状态以及表面感热、潜热和入射太阳辐射，将每步 71 维输入映射为 68 维物理倾向，并结合平流强迫执行无 teacher forcing 的多步积分。模型适用于神经网络物理参数化训练、3 小时间隔单柱预报、64 步长期稳定性验证和柱水汽收支诊断。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 单柱模式预报 | 从 34 层热力学状态和表面通量连续预测 3 小时间隔的状态演变。 |
| 物理参数化训练 | 使用 `T=20` 的多步窗口学习 71 维输入到 68 维物理倾向的映射。 |
| 长期稳定性验证 | 执行固定 64 步、共 8 日的无 teacher forcing 单柱积分。 |
| 水汽预算评估 | 结合柱水汽储量、潜热通量和平流水汽收支诊断降水。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证训练、推理、评估、可视化和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/StableNN-Phys --local_dir ./StableNN-Phys
cd StableNN-Phys
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

训练数据为大气单柱温湿状态和外部物理强迫。输入包含 34 层静力能、34 层总水以及表面通量和太阳辐射，共 71 维。目标为同一大气柱的 34 层温度和湿度物理倾向，共 68 维。样本按 3 小时间隔组成 `T=20` 的连续窗口。本仓库使用少量虚拟数据验证训练、推理和评估流程，不代表论文数据分布、训练规模或正式性能。

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
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练使用 Adam 拟合 `T=20` 的多步状态序列，默认 `paper` 损失为逐层质量加权 MAD。论文配置记录为学习率 0.01、批量 200、5 epoch 和隐藏层 128，默认工程配置缩小隐藏层与样本规模以快速验证流程，结果保存到：

```text
result/checkpoints/stablenn_phys.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置预训练权重。虚拟数据生成的 checkpoint 仅用于工程流程验证，不是论文官方权重。

### 推理

```bash
python scripts/inference.py
```

推理执行固定 64 步、共 8 日的无 teacher forcing 单柱积分，并保存每个 3 小时时刻的完整状态、倾向和强迫序列。完整数值结果保存到 `result/output/rollout.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估报告长度为 65 的逐时效质量加权 MAD 与 bias，并按数据源计算 R2。降水由柱水汽储量、潜热通量和平流水汽收支诊断，同时验证预算残差；虚拟数据结果仅验证工程流程，不代表论文正式性能。结构化结果和图片保存到：

```text
result/evaluation/metrics.json
result/evaluation/state_precipitation_timeseries.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 StableNN-Phys 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
