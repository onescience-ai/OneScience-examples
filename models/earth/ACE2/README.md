<p align="center"><strong><span style="font-size: 30px;">ACE2</span></strong></p>

# 模型介绍

ACE2 根据全球大气状态和海表温度等外部强迫，以 6 小时间隔自回归模拟天气、气候变率和长期强迫响应。模型通过硬物理校正约束干空气质量与水分收支，可用于从天气尺度延伸到长期气候统计的快速模拟。

论文：ACE2: accurately learning subseasonal to decadal atmospheric variability and forced responses  
https://doi.org/10.1038/s41612-025-01090-0

# 模型描述

该方法由艾伦人工智能研究所和美国地球物理流体动力学实验室等机构的研究团队提出。论文分别使用 ERA5 再分析和 SHiELD 历史大气模拟训练模型。ACE2 采用球面傅里叶神经算子学习全球大气状态转移，并通过硬物理校正约束干空气质量与水分收支。模型适用于 1° 全球网格上的 6 小时自回归大气模拟与气候统计分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球大气模拟 | 在 1° 网格上执行 6 小时自回归预报。 |
| 物理约束模拟 | 验证干空气质量和全球水分收支硬约束。 |
| 本地工程验证 | 使用完整网格、50 通道和8垂直层的结构化虚拟数据验证流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证数据、训练、推理、大气指标和评估流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ACE2 --local_dir ./ACE2
cd ACE2
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

本仓库使用少量虚拟样本验证工程流程，虚拟数据包含全球大气状态、外部强迫、50 个状态通道、8 个垂直层、6 小时时间关系和 `[N,T,50,180,360]` 真实空间维度。虚拟数据保持论文的网格、通道、垂直层和时间规格，仅减少样本数量、模型规模和训练轮次；50 通道采用依据正文构建的工程账本。该数据仅用于验证 SFNO、硬物理校正、训练、推理和评估流程，不代表 ERA5、SHiELD 官方数据分布与训练规模。

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
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练采用两步 6 小时自回归均方误差，并在模型输出后施加干空气质量和水分收支硬约束。默认配置缩小 SFNO 宽度、频谱模态和训练轮数，但不缩小数据维度和两步自回归目标。训练产物保存到：

```text
result/checkpoints/ace2.pt
result/training/metrics.json
```

### 训练权重

论文提供 ACE2-ERA5 官方 checkpoint：https://doi.org/10.57967/hf/5377 。本仓库不在 `weight/` 中内置该权重，紧凑工程模型不声明与官方 checkpoint 兼容。

### 推理

```bash
python scripts/inference.py
```

推理加载本地训练 checkpoint，以初始全球大气状态和逐时次外部强迫作为输入。模型按 6 小时间隔自回归生成未来 6、12 和 18 小时的大气状态，并在每一步应用物理校正。输出保持 50 通道、`180×360` 网格和时间顺序。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算纬度面积加权 RMSE、全球均值 R²、持续性基线技能，以及干空气质量和水分闭合误差。结构化结果包含模型误差、基线误差和守恒诊断，并生成模型与持续性基线误差及守恒残差对比图。虚拟数据结果仅用于工程验证，不代表论文正式性能。评估结果保存到：

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

本仓库为 ACE2 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY 4.0 许可证；论文、官方模型权重、ERA5 和 SHiELD 数据仍应按照各自项目的许可证及使用条款使用。
