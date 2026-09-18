<p align="center"><strong><span style="font-size: 30px;">PrecipDD</span></strong></p>

# 模型介绍

PrecipDD 从全球逐日降水异常图估计全球年平均近地面气温异常（AGMT），用于检测逐日降水中的人为气候变化指纹。模型以维度一致的卷积网络支持 AGMT 回归、涌现日检测、趋势分析和遮挡敏感度分析。

论文：Anthropogenic fingerprints in daily precipitation revealed by deep learning  
https://doi.org/10.1038/s41586-023-06474-x

# 模型描述

该模型由韩国蔚山科学技术院、浦项科技大学等机构的研究团队提出。模型使用 CESM2 Large Ensemble 的 80 个成员以及 1850–2100 年的逐日降水和全球年平均近地面气温数据训练。模型适用于从全球逐日降水场识别人为气候变化指纹，并估计全球年平均近地面气温异常。其核心特点是利用卷积神经网络提取降水空间分布中的升温信号，并结合涌现时间和可解释性分析定位关键区域。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| AGMT 回归 | 从单日归一化全球降水异常图估计 AGMT 异常。 |
| 涌现日检测 | 统计预测 AGMT 高于论文内部变率上界 0.42°C 的日比例。 |
| 趋势与可解释性分析 | 计算 AGMT、涌现日比例趋势和 `7×7` 遮挡敏感度趋势图。 |
| 本地工程验证 | 使用完整空间维度的结构化虚拟数据验证模型流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证数据、训练、推理和评估流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/PrecipDD --local_dir ./PrecipDD
cd PrecipDD
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本仓库使用少量虚拟样本验证工程流程，虚拟数据包含归一化逐日降水异常、AGMT 标签、年份、日序和 `55×160` 经纬网格。虚拟数据保持论文的空间维度、异常归一化方式和增暖相关信号，仅减少样本数量、集合成员数和训练轮次。该数据仅用于验证数据生成、训练、推理、涌现检测、趋势分析和遮挡评估流程，不代表 CESM2 Large Ensemble 官方数据分布与论文训练规模。

```bash
python scripts/fake_data.py
```

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=2 --nnodes=1 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练采用 Adam 优化器、MAE 损失和 L2 权重衰减，并独立训练多个随机初始化的集合成员。默认配置保留完整输入维度与网络特征尺寸，但缩减样本数量、集合成员数和训练轮次。训练产物保存到：

```text
result/checkpoints/precipdd.pt
result/training/metrics.json
```

### 训练权重

论文未提供可供本仓库直接使用的官方模型权重。本仓库不在 `weight/` 中内置权重，本地训练 checkpoint 仅用于工程流程验证。

### 推理

```bash
python scripts/inference.py
```

推理加载本地训练 checkpoint，并读取留出的逐日降水异常样本。模型对各集合成员的 AGMT 预测取平均，同时保留真值、年份、日序和网格坐标。输出仅用于后续指标与遮挡敏感度评估。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算逐日和年度 Pearson 相关系数与 RMSE，并统计预测 AGMT 高于 0.42°C 的涌现日比例。流程同时计算 AGMT 与涌现日比例趋势，并生成 `7×7` 遮挡敏感度趋势图。虚拟数据结果仅用于工程验证，不代表论文正式性能。评估结果保存到：

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

本仓库为 PrecipDD 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY 4.0 许可证；论文及 CESM2 Large Ensemble 等数据仍应按照各自项目的许可证及使用条款使用。
