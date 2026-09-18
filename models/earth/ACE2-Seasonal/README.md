<p align="center"><strong><span style="font-size: 30px;">ACE2-Seasonal</span></strong></p>

# 模型介绍

ACE2-Seasonal 复现 ACE2-ERA5 的全球季节回报试验。模型以六小时步长自回归，并通过固定 SST 与海冰异常构造 64 成员滞后集合。

论文：Skilful global seasonal predictions from a machine learning weather model trained on reanalysis data  
https://doi.org/10.1038/s41612-025-01198-3

# 模型描述

该研究由英国气象局、University of Exeter 和 Ai2 团队开展。ACE2 使用 ERA5 再分析大气场训练，季节试验使用 ERA5 初始状态和持续海洋边界异常。模型适用于全球 DJF 季节预报以及 NAO、ENSO 遥相关和集合 spread-skill 分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 季节集合预测 | 构造 64 成员滞后集合。 |
| NAO 预测 | 评估集合平均 NAO 相关和 spread。 |
| 边界强迫试验 | 保持初始化 SST 与海冰异常。 |
| ModelScope/OneCode 运行 | 验证结构化数据、训练、推理、季节指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ACE2-Seasonal --local_dir ./ACE2-Seasonal
cd ACE2-Seasonal
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

论文在全球 1° 网格上使用 ERA5 大气场，并以 6 小时步长训练 ACE2；完整通道和垂直层未在本论文列出。虚拟数据保留全球逻辑网格、边界变量、六小时时间顺序和 64 成员集合，仅将未闭合通道标记为 8 通道工程账本并减少实际 tile 与 rollout。结果仅用于工程验证。

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

训练已完成单卡和双进程 DDP 验证，并生成单一可恢复 checkpoint。六小时状态演化目标已实际完成反向传播和参数更新，并记录训练 MSE。训练结果保存到：
```text
result/checkpoints/ace2_seasonal.pt
result/training/metrics.json
```

### 训练权重

官方 ACE2-ERA5 checkpoint 可从 https://huggingface.co/allenai/ACE2-ERA5 获取，本紧凑实现不声明权重兼容。

### 推理

```bash
python scripts/inference.py
```

推理生成 64 成员六小时自回归集合并保持 SST/海冰异常，输出 shape 为 `[64,12,8,16,16]` 且数值有限。恢复后的 checkpoint 能够连续执行 12 个工程预报步，并保留全球逻辑 shape 和不完整覆盖标志。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算集合 RMSE、spread 和 NAO proxy 相关，指标和生成图均通过有效性检查。评估结果保存到：
```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 ACE2-Seasonal 公开规格的独立工程复现版本。

原始论文采用 CC BY 4.0 许可证；论文、ACE2 官方代码、权重和 ERA5 数据仍应按照各自项目的许可证及使用条款使用。
