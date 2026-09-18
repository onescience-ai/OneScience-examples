<p align="center"><strong><span style="font-size: 30px;">pysteps</span></strong></p>

# 模型介绍

pysteps 是面向概率降水临近预报的开源方法库。本复现聚焦 STEPS 的光流、级联分解、AR(2) 和集合生成流程。

论文：Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0)  
https://doi.org/10.5194/gmd-12-4185-2019

# 模型描述

该方法由 Finnish Meteorological Institute、MeteoSwiss、ETH Zurich、Colorado State University 等机构的团队提出。模型使用芬兰、瑞士、美国和澳大利亚的五分钟雷达降水序列在线估计运动场、级联和自回归参数。模型适用于未来一至三小时概率降水临近预报、阈值超越概率和集合不确定性分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 概率临近预报 | 生成 24 成员降水集合。 |
| 雷达外推 | 使用光流和半拉格朗日方法推进降水场。 |
| 阈值概率分析 | 计算不同雨强阈值的超越概率。 |
| ModelScope/OneCode 运行 | 验证结构化数据、推理、概率指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程入口。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/pysteps --local_dir ./pysteps
cd pysteps
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

论文使用多个国家和地区的五分钟雷达降水序列，典型输入为最近 3 帧，输出为未来 12 个五分钟时效。虚拟数据保持三帧输入、12 时效、8 层级联和 24 成员集合，仅缩小实际雷达网格。结果仅用于工程验证，不代表论文正式性能。

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
torchrun --standalone --nproc_per_node=2 scripts/train.py
```

STEPS 不执行离线梯度训练，`train.py` 保存在线参数估计和集合配置协议。单卡和双进程入口均已验证，结果保存到：

```text
result/checkpoints/pysteps.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。pysteps 通过最新雷达观测在线估计参数，不使用神经网络预训练权重；官方软件位于 https://github.com/pySTEPS/pysteps 。

### 推理

```bash
python scripts/inference.py
```

推理恢复在线参数协议并生成 `[24,12,128,128]` 概率集合。集合 shape 和有限数值检查均已通过，推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算逐时效 RMSE 和集合 spread，并生成时效误差图。指标与 PNG 均通过有效性检查，评估结果保存到：

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

本仓库为 pysteps 公开规格的独立工程复现版本。

原始论文采用 CC BY 4.0 许可证，官方 pysteps 软件采用 BSD-3-Clause 许可证；原始论文、官方软件和雷达数据仍应按照各自项目的许可证及使用条款使用。
