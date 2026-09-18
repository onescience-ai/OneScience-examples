<p align="center">
  <strong><span style="font-size: 30px;">PPNN</span></strong>
</p>

# 模型介绍

PPNN 用于把集合天气预报和站点信息转换为校准后的 2 米温度概率分布，主要服务于站点级集合预报后处理、预报不确定性表达和概率天气预报。

论文：Neural Networks for Postprocessing Ensemble Weather Forecasts  
https://doi.org/10.1175/MWR-D-18-0187.1

# 模型描述

该方法由 Ludwig-Maximilians-Universität München、Karlsruhe Institute of Technology 和 Heidelberg Institute for Theoretical Studies 的研究团队提出。论文使用 ECMWF TIGGE 集合预报和 Deutscher Wetterdienst 站点 2 米温度观测训练。模型适用于固定 48 小时时效的站点温度概率后处理和集合预报校准。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 集合预报后处理 | 从集合统计量和站点信息生成高斯温度概率预报。 |
| 概率预报训练 | 使用 Gaussian CRPS 训练均值和尺度参数。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证特征构造、训练、推理、校准评估和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 训练多个独立网络副本。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/PPNN --local_dir ./PPNN
cd PPNN
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
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

训练数据为 ECMWF TIGGE 集合预报和 DWD 站点 2 米温度观测。每个样本包含 50 个集合成员、18 个预报变量以及站点位置和海拔信息，目标为同一有效时刻的站点温度。预报每天 00 UTC 初始化，固定时效为 48 小时，温度单位为摄氏度。本仓库使用少量虚拟样本验证训练、推理和概率评估流程，不代表 TIGGE 或 DWD 官方数据分布、训练规模或论文正式性能。

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

默认工程配置将论文长训练模型的隐藏宽度由 512 缩为 32、独立网络数由 10 缩为 2，并减少训练轮数和样本数；50 个成员、18 个变量、537 站元数据、二维站点 embedding 和 48 小时时效没有缩小。正式实验需要 TIGGE 与 DWD 真实数据、论文规模网络集合和完整训练时段。

训练产物保存到：

```text
result/checkpoints/ppnn.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重，也未发现论文作者发布的可直接加载预训练 checkpoint。本地工程训练权重保存在 `result/checkpoints/ppnn.pt`，不得描述为官方预训练权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据 50 成员集合预报统计量和站点信息生成固定 48 小时时效的高斯温度概率预报。完整数值结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估遵循论文概率预报协议，计算 Gaussian CRPS、原始集合 CRPS、CRPSS、PIT 校准和 spread-error 诊断，并保存整体与站点样本聚合结果；任务只有固定 48 小时时效且没有类别，因此不保存逐时效或逐类别指标。评估同时生成 PIT 校准和概率预报误差对比图；虚拟数据结果仅用于工程验证，不代表论文正式性能。

```text
result/evaluation/metrics.json
result/evaluation/ppnn_evaluation.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 PPNN 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
