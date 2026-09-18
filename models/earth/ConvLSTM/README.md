<p align="center">
  <strong>
    <span style="font-size: 30px;">ConvLSTM</span>
  </strong>
</p>

# 模型介绍

ConvLSTM 是面向时空序列预测的卷积循环神经网络，将 LSTM 的输入到状态和状态到状态变换替换为空间卷积，从而在保留长期时间依赖的同时建模局部空间相关性。

论文：Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting  
https://papers.nips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html

# 模型描述

ConvLSTM 由香港科技大学与香港天文台的研究人员提出。模型使用 2011 至 2013 年香港天气雷达数据中降雨量最高的 97 天，以及 Moving-MNIST 合成序列进行训练和评估。模型适用于降水临近预报、视频预测和一般时空序列预测任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 降水临近预报 | 根据 5 张历史雷达图预测未来 15 个时间步。 |
| 时空序列建模 | 使用卷积门控结构联合学习空间和时间相关性。 |
| 多步图像预测 | 通过 Encoder-Forecaster 结构连续生成未来图像。 |
| 本地工程验证 | 使用虚拟雷达序列检查训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ConvLSTM --local_dir ./ConvLSTM
cd ConvLSTM
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

本仓库使用少量虚拟样本验证工程流程，虚拟数据包含连续 20 帧、时间间隔为 6 分钟的 `100×100` 单通道 `float32` 雷达回波图，其中前 5 帧作为输入，后 15 帧作为预测目标。该数据仅用于验证 ConvLSTM 的时空编码、15 步预测、训练、推理和评估流程，不代表官方雷达数据分布与训练规模。

```bash
python scripts/fake_data.py
```

### 训练

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

默认配置保持两层 Encoder、两层 Forecaster、`3×3` 卷积、逐通道 peephole 和完整预测长度，只缩小样本数量、隐藏通道和训练周期。

```text
result/checkpoints/convlstm.pt
result/training/metrics.json
```

### 训练权重

本仓库不内置虚拟权重或训练权重，论文未提供可直接下载的原始 Theano 预训练 checkpoint。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据 5 张历史雷达图生成未来 15 张雷达回波预测。推理结果包含输入序列、真实目标、预测序列及对应的分钟时效信息。

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估按照论文的 Z-R 关系将雷达回波转换为降雨率，并计算 Rainfall-MSE、CSI、FAR、POD 和 Correlation。结果同时包含 15 个预测时效的分步指标及整体汇总指标，并生成部分时效的目标、预测和绝对误差对比图。虚拟数据结果仅用于验证工程流程，不代表论文真实雷达数据指标。

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 ConvLSTM 论文公开规格的独立工程复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
