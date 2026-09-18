<p align="center">
  <strong><span style="font-size: 30px;">MassConservingCNN</span></strong>
</p>

# 模型介绍

MassConservingCNN 用于修正集合卡尔曼滤波数据同化过程中由局地化引起的质量守恒破坏，根据非约束分析和雷达观测位置生成满足降雨非负性并改善质量守恒的分析场，主要用于物理约束数据同化方法研究和分析场后处理。

论文：Training a convolutional neural network to conserve mass in data assimilation  
https://doi.org/10.5194/npg-28-111-2021

# 模型描述

MassConservingCNN 由 Ludwig-Maximilians-Universität München 气象研究所与 ClimateAi 的研究人员提出。论文使用一维 modified shallow-water 模型双生试验产生的 EnKF 非约束分析、QPEns 约束分析和雷达观测位置数据进行训练与验证。模型适用于质量守恒数据同化修正、降雨非负约束和物理一致分析场生成。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 数据同化修正 | 根据 `X^a` 和雷达位置指示预测 QPEns 风格分析场。 |
| 质量感知训练 | 使用论文 Equation 6 误差与 Equation 7 质量惩罚进行训练。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/MassConservingCNN --local_dir ./MassConservingCNN
cd MassConservingCNN
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

默认生成器创建 24 个训练样本和 12 个验证样本，保持 `[B,4,250]` 输入和 `[B,3,250]` 目标维度。数据包含周期波、平滑对流单体、与速度收敛相关的非负降雨、雨区 radar mask 和平滑 EnKF 风格误差，仅用于验证工程流程，不等同于论文的 48,000 样本 QPEns 数据。

```bash
python scripts/fake_data.py --force
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

训练会检查数据版本、shape、dtype、有限值、radar 二值性和降雨非负性。checkpoint 保存模型参数、优化器状态、模型配置、归一化统计量、变量顺序、数据版本、`eta`、epoch 和 seed，训练结果写入：

```text
result/checkpoints/massconservingcnn.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文未提供可确认的官方 checkpoint，当前工程 checkpoint 不声明兼容外部权重。

### 推理

```bash
python scripts/inference.py
```

推理读取训练生成的模型参数，并根据非约束分析和雷达位置指示生成质量修正后的分析场。结果包含输入分析、目标分析、模型预测、雷达观测位置以及对应的物理量和归一化信息，保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估结果包含预测误差、质量守恒误差和相对改进等核心结果，并生成输入、目标与预测的对比图。结构化指标和辅助图片分别保存到以下路径；虚拟数据结果仅用于验证工程流程，不代表论文真实性能。

```text
result/evaluation/metrics.json
result/evaluation/input_target_prediction.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 MassConservingCNN 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
