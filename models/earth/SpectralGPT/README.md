<p align="center">
  <strong>
    <span style="font-size: 30px;">SpectralGPT</span>
  </strong>
</p>

# 模型介绍

SpectralGPT 是面向光谱遥感影像的基础模型，通过空间光谱三维分块、掩码自编码和渐进式预训练学习跨波段与空间结构，可为分类、分割和变化检测等任务提供表征。

论文：SpectralGPT: Spectral Remote Sensing Foundation Model  
https://arxiv.org/abs/2311.07113

# 模型描述

SpectralGPT 由西北工业大学等机构的研究团队提出。模型先使用 `96x96` 的 fMoW-Sentinel 数据训练，再使用 `128x128` 的 BigEarthNet 数据进行第二阶段渐进式预训练。模型适用于多光谱影像重建、光谱遥感表征学习及下游遥感任务迁移。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 渐进式预训练 | 依次执行 `96x96` 第一阶段和 `128x128` 第二阶段训练。 |
| 空间光谱重建 | 对 Sentinel-2 的 12 波段空间光谱块进行掩码重建。 |
| 多光谱地物分类 | 迁移空间光谱表征并微调，用于 EuroSAT、BigEarthNet 等土地覆盖分类任务。 |
| 语义分割与变化检测 | 将预训练编码器适配到像素级地物分割和双时相遥感变化检测任务。 |
| 本地工程验证 | 使用少量虚拟数据检查训练、推理和评估流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/SpectralGPT --local_dir ./SpectralGPT
cd SpectralGPT
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

默认使用少量虚拟数据验证两阶段工程流程，第一阶段采用 fMoW-Sentinel 风格样本，第二阶段采用 BigEarthNet 风格样本。

虚拟数据保持作者官方渐进式预训练的 12 波段、第一阶段 `96x96` 和第二阶段 `128x128` 输入规格。

真实数据需预处理并转换为以下 NPZ 训练协议；该协议与模型输入规格一致，但不等同于原始数据集的下载格式。

```text
stage1:
images: float32 [N,12,96,96]
band_order: string [12]
normalization: string scalar
scale_factors: float32 [N]
stage: string scalar = stage1

stage2:
images: float32 [N,12,128,128]
band_order: string [12]
normalization: string scalar
scale_factors: float32 [N]
stage: string scalar = stage2
```

`fake_data.py` 会自动写入 `protocol` 和 `data_source` 协议元数据，使用真实数据时需保留这些字段。

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

训练先完成 96 尺寸第一阶段，再插值空间位置编码并完成 128 尺寸第二阶段，保存阶段 checkpoint、最终 checkpoint 和总体训练指标。默认配置面向快速流程验证；开展正式实验时，应使用论文对应的两阶段数据规模、模型配置和训练周期。

```text
result/checkpoints/stage1.pth
result/checkpoints/stage2.pth
result/checkpoints/final.pth
result/training/metrics.json
```

### 训练权重

本仓库将在 `weight/` 文件夹内提供 SpectralGPT 训练权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理加载最终第二阶段 checkpoint，对 `128x128` 测试数据执行掩码重建，并将结果保存到：

```text
result/output/reconstruction.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估总体报告掩码区域的 MSE、MAE、PSNR、光谱角和逐波段 RMSE，并生成输入、可见区域、预测与合成结果图。虚拟数据结果仅用于验证工程流程，不代表论文完整性能。

```text
result/output/metrics.json
result/output/reconstruction.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 SpectralGPT 原始论文的复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
