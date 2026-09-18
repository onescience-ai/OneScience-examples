<p align="center"><strong><span style="font-size: 30px;">DINCAE</span></strong></p>

# 模型介绍

DINCAE 通过概率卷积自编码器重建云遮挡造成的逐日海表温度缺测，并提供像素级重建不确定性。

论文：DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations  
https://doi.org/10.5194/gmd-13-1609-2020

# 模型描述

该方法由列日大学和斯洛文尼亚国家生物研究所的研究团队提出。论文使用 1985-2009 年 AVHRR Pathfinder 逐日海表温度资料开展训练与验证。模型适用于重建云遮挡区域的 SST，并同时估计重建误差方差。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 海表温度缺测重建 | 根据当前、前一日和后一日观测重建云遮挡 SST。 |
| 概率误差估计 | 联合输出重建均值和像素级误差方差。 |
| 本地工程验证 | 使用缩小空间尺度的结构化虚拟数据验证训练、推理和评估流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、SST 重建指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/DINCAE --local_dir ./DINCAE
cd DINCAE
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 训练数据介绍

```bash
python scripts/fake_data.py
```

论文数据包含 5266 个 `112×112` AVHRR 逐日 SST 时刻，模型输入当前、前一日和后一日观测及精度、坐标和季节信息。本仓库默认使用 8 个 `32×32` 结构化虚拟时刻以控制 CPU 验证成本，论文真实网格和模型参数单独记录在 `paper_model`。虚拟数据仅用于验证工程流程，不代表 AVHRR 的真实数据分布、训练规模或论文性能。

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练使用随机云掩码和 masked Gaussian NLL；默认缩小网格、样本数、卷积宽度、瓶颈和训练轮数，论文配置保留 `112×112` 网格与 1000 epochs。正式实验需要完整 AVHRR 数据和论文规模模型，训练产物保存到：

```text
result/checkpoints/dincae.pt
result/training/metrics.json
```

### 训练权重

论文未提供可直接加载的官方模型权重，本仓库不在 `weight/` 中内置权重。本地训练生成的 checkpoint 保存到 `result/checkpoints/dincae.pt`，不得描述为官方预训练权重。

### 推理

```bash
python scripts/inference.py
```

输出 `result/output/predictions.npz`，包含重建均值、方差、目标、缺测掩膜和日期。

### 评估和可视化

```bash
python scripts/result.py
```

评估输出 RMSE、中心化 RMSE（CRMSE）、bias、标准化残差 calibration，以及 DINEOF-like rank-13 迭代低秩基线。结果写入 `result/evaluation/metrics.json` 和 `result/evaluation/comparison.png`。少样本指标只证明流程连通，不代表论文性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 DINCAE 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
