<p align="center">
  <strong>
    <span style="font-size: 30px;">Prithvi-EO</span>
  </strong>
</p>

# 模型介绍

Prithvi-EO-2.0 是面向多时相地球观测数据的基础模型，将 HLS 多光谱时间序列、影像获取日期和地理位置编码为统一表征，并通过 Masked Autoencoder 重建被遮挡的时空 Patch，可用于遥感分类、语义分割、回归和环境变化监测。

论文：Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications  
https://arxiv.org/abs/2412.02732

# 模型描述

Prithvi-EO-2.0 由 IBM、NASA 和 Jülich Supercomputing Centre 联合提出。模型使用 NASA Harmonized Landsat Sentinel-2 数据集中 420 万个全球四时相样本训练。模型适用于灾害响应、土地覆盖与作物制图、生态系统动态监测，以及遥感影像分类、分割和回归任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多时相表征 | 使用 Transformer 同时编码四个时间步的空间和时间信息。 |
| 遥感影像重建 | 使用 3D Masked Autoencoder 重建被遮挡的多光谱时空 Patch。 |
| 时空元数据建模 | 融合年份、年积日、纬度和经度，并支持训练时随机丢弃元数据。 |
| 本地工程验证 | 使用少量虚拟 HLS 样本检查训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/Prithvi-EO --local_dir ./PrithviEO
cd PrithviEO
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证，官方尺寸模型训练需要大规模加速资源。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本仓库使用少量虚拟样本验证工程流程，训练数据和测试数据分别保存为 `data/train.npz` 和 `data/test.npz`。虚拟数据保持论文的四时间步与六个 HLS 公共波段，波段顺序为 Blue、Green、Red、Narrow NIR、SWIR1 和 SWIR2，并使用官方公开的均值与标准差完成归一化。

虚拟数据保持论文训练使用的 4×224×224 时空尺寸，每个样本的完整影像张量为 6×4×224×224。当前仅缩小样本数量、模型宽度、模型深度和训练周期，用于验证 3D Patch Embedding、时空位置编码、时间与位置元数据编码和 MAE 训练流程，不代表官方 420 万个 HLS 样本的数据分布与训练规模。

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

默认配置面向快速流程验证；开展正式实验时，应使用真实 HLS 时间序列、官方 300M 或 600M 配置和完整训练周期。

```text
result/checkpoints/prithvi_eo.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置虚拟权重或官方权重。IBM 和 NASA 已在 Hugging Face 公开 Prithvi-EO-2.0 的 tiny、100M、300M 和 600M 权重，包括带时间与位置编码的 TL 版本：

https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL

本仓库是缩小的独立工程实现，模型参数名称和尺寸不与官方权重兼容。需要使用官方权重时，应采用 TerraTorch 或官方仓库提供的实现和数据预处理流程。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，生成 CLS embedding、时空 Patch embedding、被遮挡 Patch 的多时相重建结果和 Mask，并保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估模型的 masked-patch MSE、完整时空重建 MAE、各时间步重建误差和 embedding 范数。评估过程同时生成四个时间步的输入影像、重建结果和绝对误差对比图。虚拟数据结果仅用于验证工程流程，不代表论文中的 GEO-Bench 或真实下游任务性能。

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

本仓库为 Prithvi-EO-2.0 论文公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
