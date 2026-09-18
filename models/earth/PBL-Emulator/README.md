<p align="center">
  <strong><span style="font-size: 30px;">PBL-Emulator</span></strong>
</p>

# 模型介绍

PBL-Emulator 使用领域感知神经网络，根据近地面状态与强迫变量离线诊断同一时刻的行星边界层风、温度和水汽垂直剖面。

论文：Fast domain-aware neural network emulation of a planetary boundary layer parameterization in a numerical weather forecast model  
https://doi.org/10.5194/gmd-12-4261-2019

# 模型描述

该方法由 Argonne National Laboratory 环境科学部、数学与计算机科学部的研究团队提出。
论文使用 WRF v3.3.1 在 NCEP-R2 驱动下采用 YSU 行星边界层方案生成的 1984-2005 年数据。
模型适用于离线诊断同一时刻的 PBL 垂直剖面，不用于未来时刻预报。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 离线 PBL 剖面诊断 | 根据已有近地面状态与强迫变量诊断同一时刻的风、温度和水汽剖面。 |
| HPC/HAC 垂向依赖研究 | 验证 HPC 的相邻低层条件依赖和 HAC 的全部低层条件依赖。 |
| 工程验证 | 使用结构化虚拟数据验证数据生成、训练、推理、评估和可视化流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、PBL 剖面指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/PBL-Emulator --local_dir ./PBL-Emulator
cd PBL-Emulator
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的流程验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
pip install numpy pyyaml matplotlib
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
pip install numpy pyyaml matplotlib
```

### 训练数据介绍

每个样本由 16 个近地面状态和强迫变量输入映射到同一时刻 17 个垂直位置上的 5 个变量输出，数据形状为 `[N,16]` 到 `[N,17,5]`。结构化虚拟数据包含昼夜和季节变化、垂向结构以及热量、湿度和风场等物理关联。这些数据仅用于工程验证，不代表 WRF 的数据分布、数据规模或论文性能。

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

默认配置只将训练轮数从论文协议的 1000 epochs 缩小为 6 epochs，不缩小输入或输出维度。正式实验应使用 1984-2005 年 WRF/NCEP-R2/YSU 数据和完整训练轮数，训练产物保存到：

```text
result/checkpoints/pbl_emulator.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置论文官方权重，也未发现论文作者发布的可直接加载预训练 checkpoint。本地训练生成的 `result/checkpoints/pbl_emulator.pt` 是当前数据对应的工程 checkpoint，不得描述为官方预训练权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，为测试样本生成同一时刻的 PBL 垂直剖面诊断结果。完整数值结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估提供分变量误差和相关性，并绘制目标与预测剖面图。虚拟数据结果仅用于工程验证，不代表论文性能，结果保存到：

```text
result/evaluation/metrics.json
result/evaluation/profiles.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 PBL-Emulator 公开规格的独立工程复现版本；论文文本采用 CC BY 4.0，论文官方代码采用 BSD-3-Clause。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
