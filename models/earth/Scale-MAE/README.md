<p align="center"><strong><span style="font-size: 30px;">Scale-MAE</span></strong></p>

# 模型介绍

Scale-MAE 是面向多尺度地理空间影像的尺度感知掩码自编码器，通过地面采样距离感知的位置编码、可见块编码和低高频目标重建学习稳定的遥感图像表示。

论文：Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning  
https://arxiv.org/abs/2212.14532

# 模型描述

Scale-MAE 由美国国家航空航天局喷气推进实验室与斯坦福大学研究团队提出。模型使用 FMoW-RGB 等多尺度地理空间影像进行训练。模型适用于遥感图像表征学习、场景分类和建筑物分割等任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多尺度遥感预训练 | 使用带 GSD 元数据的配对低分辨率和高分辨率 `BCHW` 影像。 |
| 场景分类 | 通过可复用 CLS 特征执行 kNN 迁移评估。 |
| 建筑物分割 | 将尺度感知表征迁移到 SpaceNet 等建筑物语义分割任务并进行微调。 |
| 低高频重建 | 使用面积重采样和带通目标评估尺度敏感性。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、训练、推理和评估。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于当前默认小配置的流程验证。
- DCU 用户需要预先安装与集群匹配的 DTK，建议使用 DTK 25.04.2 以上版本。

### 下载模型包

```bash
modelscope download --model OneScience/Scale-MAE --local_dir ./Scale-MAE
cd Scale-MAE
```

### 安装运行环境

**DCU环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

论文使用 FMoW-RGB 等多尺度地理空间影像进行预训练，并在 RESISC-45、UCMerced、EuroSAT、AID、MLRSNet 和 SpaceNet 等任务上评估。数据文件包含 `images` `[B,C,H,W]`、`targets` `[B,C,Ht,Wt]`、`gsd` `[B]` 和 `labels` `[B]`。

默认使用虚拟数据：

```bash
python scripts/fake_data.py
```

使用真实数据时，不要运行 `fake_data.py`。请先将数据整理为以下目录和字段，并替换 `data/` 下由虚拟数据脚本生成的文件：

```text
data/train.npz
data/test.npz
```

每个 NPZ 文件至少包含：

```text
images:  float32 [N,C,input_size,input_size]
targets: float32 [N,C,target_size,target_size]
gsd:     float32 [N]
labels:  int64   [N]
```

其中 `images` 是模型输入，`targets` 是与输入场景对应的目标分辨率影像，`gsd` 是每个样本的地面采样距离（米/像素），`labels` 用于 kNN 特征评估。真实数据的通道数、输入尺寸、目标尺寸和 GSD 范围必须与 `conf/config.yaml` 及模型配置一致；数据应在生成 NPZ 前完成裁剪、配准、通道整理和数值归一化。

根据真实数据修改 `conf/config.yaml` 中的 `input_size`、`target_size`、`channels`、`gsd_values` 和路径。完成数据准备后，继续使用下面统一的训练、推理和评估命令；如需使用其他文件位置，再通过脚本参数覆盖默认路径。

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 scripts/train.py
```

训练输出：

```text
result/checkpoints/scalemae.pt
result/training/metrics.json
```

训练输出包括可用于后续推理和特征提取的模型检查点，以及反映整体、低频和高频重建损失变化的训练指标，便于保存训练状态并分析模型优化效果。

AdamW 使用 betas `(0.9, 0.95)`，包含梯度累积、AMP、warmup 和余弦衰减。

### 训练权重

本仓库在 `weight/` 文件夹内提供基于多尺度地理空间影像训练的权重，权重文件即将上传。

### 推理

```bash
python scripts/inference.py
```

推理结果输出到：

```text
result/output/reconstruction.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估和可视化输出到：

```text
result/evaluation/metrics.json
result/evaluation/features.npy
result/evaluation/bandpass_reconstruction.png
result/evaluation/frequency_error.png
result/evaluation/gsd_reconstruction_error.png
result/evaluation/gsd_knn_accuracy.png
```

评估结果综合反映整体及低高频重建质量、不同 GSD 下的尺度适应性和表征特征的 kNN 分类能力，并通过重建对比与尺度变化曲线展示模型对多尺度地理空间影像的处理效果。当前结果基于少量虚拟数据，主要用于确认训练、推理、评估和可视化流程正常运行。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 Scale-MAE 原始论文的复现版本。
