<p align="center">
  <strong>
    <span style="font-size: 30px;">SmaAt-UNet</span>
  </strong>
</p>

# 模型介绍

SmaAt-UNet 是面向降水临近预报的轻量卷积神经网络，将连续雷达降水图作为输入，预测未来多个时间步的降水分布。模型在 U-Net 中加入卷积块注意力模块和深度可分离卷积，以较少参数保持具有竞争力的预报性能。

论文：SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture  
https://arxiv.org/abs/2007.04417

# 模型描述

SmaAt-UNet 由 Maastricht University 研究人员提出。模型使用荷兰 KNMI 2016 至 2019 年约 42 万张 5 分钟间隔雷达降水图，并在法国二值云覆盖数据上进行评估。模型适用于短时降水临近预报、云覆盖预测和轻量气象图像回归任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 降水临近预报 | 根据过去 60 分钟雷达图预测未来 30 分钟降水。 |
| 多步图像回归 | 一次输出 6 张连续的降水预测图。 |
| 注意力特征提取 | 使用 CBAM 强化重要通道和空间区域。 |
| 本地工程验证 | 使用虚拟雷达序列检查训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/SmaAt-UNet --local_dir ./SmaAtUNet
cd SmaAtUNet
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

本仓库使用归一化虚拟雷达序列验证工程流程。每个样本保持官方降水任务的真实数据维度：输入为 `12×288×288`，目标为 `6×288×288`，相邻图像间隔 5 分钟。

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

默认配置只缩小样本数量、基础特征宽度和训练周期，不改变输入帧数、输出帧数或空间网格尺寸。

```text
result/checkpoints/smaat_unet.pt
result/training/metrics.json
```

### 训练权重

本仓库不内置虚拟权重或训练权重，作者目前未公开可直接下载的预训练 checkpoint。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据 12 张历史降水图生成未来 6 个时间步的降水预测及注意力图，并保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估 MSE、MAE、Precision、Recall、F1、CSI、FAR 和 HSS，并生成未来 30 分钟目标、预测和绝对误差对比图。虚拟数据结果仅用于验证工程流程，不代表论文真实雷达测试集指标。

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

本仓库为 SmaAt-UNet 论文公开规格的独立工程复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
