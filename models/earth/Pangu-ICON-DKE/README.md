<p align="center">
  <strong><span style="font-size: 30px;">Pangu-ICON-DKE</span></strong>
</p>

# 模型介绍

Pangu-ICON-DKE 实现 Pangu-Weather 与 ICON 集合预报扰动增长研究中的无训练评估协议，通过差异动能（DKE）地图、全球平均和总波数谱诊断扰动的逐小时演变。

论文：Can Artificial Intelligence-Based Weather Prediction Models Simulate the Butterfly Effect?  
https://doi.org/10.1029/2023GL105747

# 模型描述

Pangu-ICON-DKE 对应的研究由德国航空航天中心（DLR）与慕尼黑大学（LMU）的研究团队提出。论文使用 5 个实验、每个实验 5 个成员的 73 小时集合预报数据，分析 300 hPa、`721×1440` 全球网格上的场和 T719 谱。该任务无需模型训练，用于比较 Pangu-Weather 与 ICON 的初值扰动增长、蝴蝶效应和内禀可预报性。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 蝴蝶效应评估 | 比较 Pangu-Weather 与 ICON 的初值扰动增长。 |
| DKE 与谱诊断 | 计算全球 DKE 地图、逐小时增长和 T719 谱。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证虚拟数据生成、推理、评估和可视化流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证完整无训练评估流程。 |
| 多卡协议检查 | 通过 `torchrun` 对五个实验执行无重复分片检查。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/Pangu-ICON-DKE --local_dir ./Pangu-ICON-DKE
cd Pangu-ICON-DKE
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认虚拟数据配置的连通性和协议检查。
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

本仓库使用参数化虚拟数据验证工程流程，保留 5 个实验、每个实验 5 个成员、73 个逐小时时次、300 hPa 单层、`721×1440` 全球网格和 T719 谱协议。数据通过惰性分块生成，以解析参数替代真实预报而不落盘完整原始场。虚拟数据仅用于验证无训练评估、推理和可视化流程，不代表 Pangu、ICON 或 ECMWF 官方数据及论文结果。

```bash
python scripts/fake_data.py --force
```

### 训练

单卡协议检查可使用：

```bash
python scripts/train.py
```

8 卡协议检查可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

论文没有模型训练步骤，`train.py` 仅检查数据协议、实验分片和零参数 DKE 前向诊断，不创建优化器、不执行反向传播，也不产生训练所得参数。协议检查产物保存到：

```text
weight/pangu_icon_dke.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置官方权重。该论文是无训练评估研究，协议检查生成的 checkpoint 仅保存配置与协议状态，Pangu 官方权重和 ICON 软件不随本仓库分发。

### 推理

```bash
python scripts/inference.py
```

推理生成五组实验的全球 DKE、逐时次空间场和 T719 谱，产物保存到：

```text
result/inference_results.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估逐一覆盖全部 73 个时次，报告 DKE、逐时增长、缩放补偿和实验间空间相关；该任务没有类别，因此不生成逐类别指标。评估生成 DKE 时间曲线、72 小时全球地图和 T719 谱图。虚拟数据结果仅用于工程流程验证，不代表论文正式性能，产物保存到：

```text
result/evaluation/metrics.json
result/evaluation/dke_diagnostics.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 Pangu-ICON-DKE 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
