<p align="center"><strong><span style="font-size: 30px;">Streamflow-LSTM</span></strong></p>

# 模型介绍

Streamflow-LSTM 是 Hunt 等人提出的逐站 LSTM 河流流量预报方法的工程复现。模型根据过去 7 天的六小时气象与水文序列，为美国西部 10 个站点生成未来 10 天、共 40 个六小时时效的流量预报。

论文：Using a long short-term memory (LSTM) neural network to boost river streamflow forecasts over the western United States  
https://doi.org/10.5194/hess-26-5449-2022

# 模型描述

该模型由雷丁大学、欧洲中期天气预报中心、拉夫堡大学和英国生态与水文中心的研究团队提出。模型使用美国西部 10 个测站的流量观测，以及流域平均气象预报和水文历史序列训练。模型适用于提升美国西部河流未来 10 天的逐站流量预报，并可与 persistence 和 GloFAS 等基线比较。其核心特点是为每个测站独立学习时序关系，并通过成员筛选和集合平均提高预报稳定性。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 逐站流量预报 | 为 10 个指定站点生成 40 个六小时时效的流量序列。 |
| 水文集合实验 | 按验证 NSE 选择不同随机初始化的站点成员并执行集合平均。 |
| 本地工程验证 | 使用保持 `10×28×23×40` 核心协议的结构化虚拟数据检查完整流程。 |
| ModelScope/OneCode 运行 | 验证数据生成、训练、推理、评估和可视化入口。 |
| 多卡训练 | 使用 `torchrun` 分配站点成员任务并汇总单一 checkpoint。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/Streamflow-LSTM --local_dir ./Streamflow-LSTM
cd Streamflow-LSTM
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行论文规模集合。
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

论文数据由 ERA5、IFS、GloFAS 和 USGS 等来源构建，覆盖 10 个站点，并将过去 7 天整理为 28 个六小时步、每步 23 个变量。仓库内的确定性虚拟数据保持 10 站、`[B,28,23]` 输入和 40 个六小时时效，仅缩减训练、验证和预报样本数。虚拟数据只用于验证工程流程，不代表官方数据分布、论文训练规模或论文性能，且预报输入在起报后冻结观测流量以避免未来观测泄漏。

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

训练对每个站点成员使用 MSE、Adam、`0.001` 学习率和 `0.1` dropout，并由各进程分担成员任务。传入 `--paper` 可恢复 50 隐藏单元、每站 100 个成员和最佳 5 个成员协议。全部站点和成员整合到单一 checkpoint，产物为：

```text
result/checkpoints/streamflow_lstm.pt
result/training/metrics.json
```

### 训练权重

论文没有发布可直接加载的官方权重，`weight/` 仅提供状态说明。本地训练会将 10 个站点的全部集合成员、归一化统计量和验证 NSE 保存在单一 `result/checkpoints/streamflow_lstm.pt` 中，不应将其描述为官方预训练权重。

### 推理

```bash
python scripts/inference.py
```

推理从单一 checkpoint 恢复全部站点成员，并校验格式版本及 10 个站点的顺序。每站按验证 NSE 选择默认最佳 2 个或论文模式最佳 5 个成员，对 `[C,40,28,23]` 输入分别计算并进行集合平均。输出裁剪为非负流量，保持 40 个六小时时效和 `m3 s-1` 单位。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估逐站计算 2、5、8 天时效的 KGE、NSE 和 RMSE，并给出 KGE 的相关、变异与偏差分量。模型结果同时与起报时流量持续性基线和合成 GloFAS proxy 对比。可视化展示全时效平均 RMSE 以及 5 天逐站 KGE，虚拟数据结果只用于工程验证。评估产物保存到：

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

本仓库为 Streamflow-LSTM 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY 4.0 许可证；论文、模型权重以及 ERA5、IFS、GloFAS 和 USGS 数据仍应按照各自项目的许可证及使用条款使用。
