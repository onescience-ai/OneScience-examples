<p align="center">
  <strong>
    <span style="font-size: 30px;">Clay Foundation Model</span>
  </strong>
</p>

# 模型介绍

Clay Foundation Model 是 Clay v1.5 公开规格的独立工程复现模型，将不同传感器的卫星影像、波段中心波长、地面采样距离、时间和经纬度统一编码为地球观测 embedding，并通过 Masked Autoencoder 重建输入波段，可作为地物分类、回归、变化检测和其他下游遥感任务的表征骨干网络。

官方文档：Clay Foundation Model v1.5  
https://clay-foundation.github.io/model/release-notes/specification.html

# 模型描述

Clay Foundation Model 由 Clay Foundation 提出。模型使用 Sentinel-2、Landsat、Sentinel-1、NAIP、LINZ 和 MODIS 等多传感器地球观测数据训练。模型适用于遥感影像表征、多波段重建、地物分类、回归和变化检测等任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多传感器表征 | 使用同一模型处理 Sentinel-2、Landsat 和 Sentinel-1 等不同波段数量的影像。 |
| 动态光谱编码 | 根据输入波段的中心波长动态生成 Patch Embedding，验证新传感器接入方式。 |
| 遥感影像重建 | 使用 Masked Autoencoder 对被遮挡的多波段影像 Patch 进行重建。 |
| 本地工程验证 | 使用少量虚拟数据检查训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ClayFoundation --local_dir ./ClayFoundation
cd ClayFoundation
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

本仓库使用少量虚拟样本验证工程流程，训练数据和测试数据分别保存为 `data/train.npz` 和 `data/test.npz`。虚拟数据包含 Sentinel-2 10 波段、Landsat 6 波段和 Sentinel-1 2 波段影像，并为每个传感器提供官方元数据对应的中心波长和地面采样距离。样本同时包含 week、hour、latitude、longitude 元数据、合成 teacher 表征目标，以及用于轻量评估的分类和回归目标。

默认输入尺寸为 64×64，训练集包含 4 个样本，测试集包含 2 个样本。该规模只用于验证动态波段接口、时空元数据编码、MAE 训练和多传感器推理流程，不代表官方约 7000 万全球遥感 chips 的数据分布与训练规模。

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

默认配置面向快速流程验证；开展正式实验时，应使用真实遥感数据、官方尺寸配置、DINOv2 teacher 和完整训练周期。

```text
result/checkpoints/clayfoundation.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置虚拟权重或官方权重。Clay Foundation 已公开 Clay v1.5 官方 checkpoint：

https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt

本仓库是缩小的独立工程实现，模型参数名称和尺寸不与官方约 1.25 GB Encoder 权重兼容。需要使用官方权重时，应采用 Clay 官方仓库提供的模型实现和数据预处理流程。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，分别处理 Sentinel-2、Landsat 和 Sentinel-1 测试影像，生成 Encoder embedding、L2 归一化投影 embedding、完整多波段 MAE 重建结果，以及重建损失和 teacher 表征对齐损失，并保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估模型在不同传感器上的影像重建与表征效果，并统计重建误差、embedding 范数和表征相似度。评估过程同时生成输入影像、重建结果和绝对误差对比图。虚拟数据结果仅用于验证工程流程，不代表官方 Clay v1.5 的训练损失、真实下游任务性能或跨区域泛化能力。

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

本仓库为 Clay Foundation Model v1.5 公开规格的独立工程复现版本。Clay 官方源码和模型权重采用 Apache-2.0 许可证。

Clay 官方仓库：  
https://github.com/Clay-foundation/model

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
