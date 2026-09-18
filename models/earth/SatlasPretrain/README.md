<p align="center">
  <strong>
    <span style="font-size: 30px;">SatlasPretrain</span>
  </strong>
</p>

# 模型介绍

SatlasPretrain 是面向大规模遥感理解的多任务预训练模型，通过双路层次骨干处理多时相高分辨率 RGB 与 Sentinel-2 多光谱影像，并联合学习稠密预测和全局分类任务。

论文：SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding  
https://arxiv.org/abs/2211.15660

# 模型描述

SatlasPretrain 由 Allen Institute for AI 研究团队提出。模型使用 SatlasPretrain 中的 NAIP 风格高分辨率影像、Sentinel-2 多光谱影像和多类型遥感标签训练。模型适用于语义分割、回归、点线面目标预测、属性识别和场景分类等遥感多任务学习。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多时相遥感融合 | 联合处理 4 时相高分辨率 RGB 与 8 时相 Sentinel-2 数据。 |
| 遥感多任务学习 | 同时预测分割、回归、点、面、线、属性和分类任务。 |
| 土地覆盖与作物制图 | 利用多时相影像执行土地覆盖、作物类型等语义分割任务。 |
| 基础设施要素提取 | 识别建筑物、道路、铁路、机场和能源设施等点线面地理目标。 |
| 本地工程验证 | 使用少量虚拟数据检查训练、推理和评估流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/SatlasPretrain --local_dir ./SatlasPretrain
cd SatlasPretrain
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

默认使用 1 个训练和 1 个测试虚拟样本验证工程流程，分别保存为 `data/train.npz` 和 `data/test.npz`。

虚拟数据保持作者官方多时相配置的 4 时相 NAIP、8 时相 Sentinel-2、`512x512` 空间尺寸和七类任务标签规格。

真实数据需预处理并转换为以下 NPZ 训练协议；该协议与模型输入规格一致，但不等同于原始数据集的下载格式。

```text
highres_images: float32 [N,4,3,512,512]
lowres_images: float32 [N,8,9,512,512]
valid_highres_times: bool [N,4]
valid_lowres_times: bool [N,8]
sample_ids: string [N]
segmentation: int64 [N,512,512]
regression: float32 [N,1,512,512]
point: float32 [N,1,512,512]
polygon: float32 [N,1,512,512]
polyline: float32 [N,1,512,512]
property: int64 [N]
classification: int64 [N]
```

`fake_data.py` 会自动写入 `protocol` 和 `source` 协议元数据，使用真实数据时需保留这些字段。

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

训练联合优化七类遥感任务，并保存 checkpoint 与总体训练指标。默认配置面向快速流程验证；开展正式实验时，应使用论文对应的多时相数据、完整任务标签、模型配置和训练周期。

```text
result/checkpoints/satlaspretrain.pt
result/training/metrics.json
```

### 训练权重

本仓库将在 `weight/` 文件夹内提供 SatlasPretrain 训练权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，生成七类任务预测并保留样本身份和协议元数据，结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估总体覆盖分割、回归、点线面目标、属性和分类任务，并生成多任务预测图。虚拟数据结果仅用于验证工程流程，不代表论文完整性能。

```text
result/evaluation/metrics.json
result/evaluation/multitask_predictions.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 SatlasPretrain 原始论文的复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
