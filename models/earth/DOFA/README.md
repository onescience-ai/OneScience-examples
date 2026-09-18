<p align="center">
  <strong>
    <span style="font-size: 30px;">DOFA</span>
  </strong>
</p>

# 模型介绍

DOFA 是面向多传感器遥感影像的动态单骨干基础模型，通过波长条件超网络生成波段自适应权重，使同一模型能够处理不同通道数和光谱响应的观测数据。

论文：Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation  
https://arxiv.org/abs/2403.15356

# 模型描述

DOFA 由武汉大学等机构的研究团队提出。模型使用 Sentinel-1、Sentinel-2、NAIP、EnMAP 和 Gaofen 五种模态进行多传感器掩码预训练，其中 Gaofen 输入为 4 通道。模型适用于多模态遥感表征学习、影像重建及跨传感器下游任务适配。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多传感器预训练 | 使用中心波长驱动动态 patch embedding 和解码权重。 |
| 跨模态表征学习 | 同一 checkpoint 处理不同通道数的五种遥感模态。 |
| 地物与场景分类 | 迁移共享表征并微调，用于不同传感器下的土地覆盖和遥感场景分类。 |
| 语义分割 | 将波长感知特征迁移到洪水、土地覆盖等像素级遥感解译任务并进行微调。 |
| 本地工程验证 | 使用少量虚拟数据检查训练、推理和评估流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/DOFA --local_dir ./DOFA
cd DOFA
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于小配置连通性验证，完整训练和推理速度较慢。
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

默认使用每种模态每个数据划分 1 个 `224x224` 虚拟样本验证工程流程。五种模态为 Sentinel-1 2 通道、Sentinel-2 9 通道、NAIP 3 通道、EnMAP 202 通道和 Gaofen 4 通道，虚拟 EnMAP 波长为协议验证使用的等间隔近似值。

虚拟数据保持作者官方预训练配置的五种传感器通道数量、`224x224` 空间尺寸和逐通道波长输入规格。

真实数据需预处理并转换为以下 NPZ 训练协议；该协议与模型输入规格一致，但不等同于原始数据集的下载格式。

```text
images: float32 [N,C,224,224]
wavelengths: float32 [C]
modality: string scalar
data_range: float scalar
```

`fake_data.py` 会自动写入 `protocol`、`data_source` 和 `wavelength_mode` 协议元数据，使用真实数据时需保留这些字段。

```bash
python scripts/fake_data.py
```

### 训练

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 scripts/train.py
```

训练在五种模态间共享动态骨干并保存 checkpoint 与总体训练指标。默认配置面向快速流程验证；开展正式实验时，应使用论文对应的多模态数据规模、模型配置和训练周期。

```text
result/checkpoints/dofa.pt
result/training/metrics.json
```

### 训练权重

本仓库将在 `weight/` 文件夹内提供 DOFA 训练权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，对配置中的各模态执行掩码重建，并将结果保存到：

```text
result/output/
```

### 评估和可视化

```bash
python scripts/result.py
```

评估总体报告掩码区域的 MSE、MAE、PSNR 和掩码比例，并生成各模态重建对比图。虚拟数据结果仅用于验证工程流程，不代表论文完整性能。

```text
result/evaluation/metrics.json
result/evaluation/
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 DOFA 原始论文的复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
