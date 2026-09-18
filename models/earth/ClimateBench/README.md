<p align="center">
  <strong><span style="font-size: 30px;">ClimateBench CNN-LSTM</span></strong>
</p>

# 模型介绍

ClimateBench CNN-LSTM 用于解决地球系统模式计算成本高、难以快速比较大量排放情景的问题，根据温室气体和气溶胶排放估计全球气温、昼夜温差和降水响应。模型主要用于数据驱动气候投影、不同排放情景的快速评估以及气候模拟方法的标准化比较，为研究气候变化响应提供高效基线。

论文：ClimateBench v1.0: A Benchmark for Data-Driven Climate Projections  
https://doi.org/10.1029/2021MS002954

# 模型描述

ClimateBench 由 University of Oxford 牵头，联合 North Carolina State University、Norwegian Meteorological Institute、University of East Anglia、Universitat de València 等机构的研究团队提出。论文使用 CMIP6、ScenarioMIP、AerChemMIP 和 DAMIP 中 NorESM2-LM 的人为强迫与气候响应数据训练和验证基线模型。模型适用于全球空间气候响应模拟、数据驱动气候投影和 SSP 情景评估。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 气候响应模拟 | 从 10 年人为强迫序列预测四类全球空间响应。 |
| 架构复现 | 验证论文 CNN、池化、ReLU LSTM 和稠密输出的精确参数化。 |
| SSP245 评估 | 按 2080-2100 目标语义计算 ClimateBench NRMSE。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 使用 `torchrun` 和 DDP 训练四个独立分支。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ClimateBench --local_dir ./ClimateBench
cd ClimateBench
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

本仓库使用少量结构化虚拟样本验证工程流程，输入包含连续年度的累积 CO2、CH4、SO2 和黑碳排放，目标为相应的全球气温、昼夜温差及降水响应。虚拟数据保留论文的时间、通道和全球网格维度，并包含合理的时间变化与空间气候结构，仅用于验证模型训练、推理和评估流程，不代表官方数据分布与论文性能。

```bash
python scripts/fake_data.py --force
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

训练输出保存到：

```text
result/checkpoints/climatebench.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文未提供可确认的官方 checkpoint，当前工程 checkpoint 不声明兼容外部权重。

### 推理

```bash
python scripts/inference.py
```

推理输出 `result/output/predictions.npz`，保留四目标预测、真实目标、坐标、场景和 2080-2100 评估语义。

### 评估和可视化

```bash
python scripts/result.py
```

评估计算四个气候变量的空间误差、全球平均误差和综合 NRMSE，并保存到 `result/evaluation/metrics.json`。脚本同时生成目标、预测和误差对比图 `result/evaluation/four_targets.png`。虚拟数据结果仅用于验证工程流程，不代表论文真实测试集性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 ClimateBench 论文公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
