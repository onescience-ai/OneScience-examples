<p align="center"><strong><span style="font-size: 30px;">WRF-ML</span></strong></p>

# 模型介绍

WRF-ML 是连接 WRF v4.3 与 Python 机器学习参数化的耦合框架。本复现聚焦论文推荐的 Model D，用双向 LSTM 模拟 RRTMG 短波和长波辐射输出。

论文：WRF–ML v1.0: a bridge between WRF v4.3 and machine learning parameterizations and its application to atmospheric radiative transfer  
https://doi.org/10.5194/gmd-16-199-2023

# 模型描述

该模型由阿里巴巴达摩院团队提出。模型使用 WRF 运行生成的 RRTMG 输入与输出大气柱数据训练，每个样本包含 57 个垂直层。模型通过布局预处理和同步请求-返回耦合协议，适用于短波和长波通量及加热率模拟以及 WRF 在线物理参数化。

# 适用场景

| 场景 | 说明 |
|---|---|
| 辐射仿真 | 模拟短波、长波通量和加热率。 |
| WRF 参数化 | 验证大气柱批处理和结果布局恢复。 |
| 离线评估 | 计算六类辐射输出 RMSE。 |
| 耦合协议 | 演示同步阻塞请求、推理和返回。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、概率降水指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/WRF-ML --local_dir ./WRF-ML
cd WRF-ML
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

论文使用 `190×170` 水平网格、57 个垂直层和每 30 分钟保存的 WRF-RRTMG 大气柱数据。虚拟数据保持 57 层、WRF `(i,k,j)` 布局转换和六类输出语义，仅减少大气柱样本、隐藏宽度和训练轮数；由于正文未列出全部输入变量，10 个输入特征明确标记为工程账本。结果只验证工程流程，不代表论文性能。

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

默认虚拟数据训练能够完成 57 层辐射剖面和边界通量的联合优化，单卡与双进程 DDP 流程均已验证通过。训练生成可恢复的单一 Model D 风格 checkpoint，并记录归一化均方误差。训练结果保存到：
```text
result/checkpoints/wrf_ml.pt
result/training/metrics.json
```

### 训练权重

论文代码与数据存档位于 https://doi.org/10.5281/zenodo.7407487 ，正文未确认可独立加载的官方预训练权重许可证，因此不提供权重链接。

### 推理

```bash
python scripts/inference.py
```

推理恢复 checkpoint，通过布局适配器执行同步耦合调用，并输出 57 层剖面和边界通量。剖面输出维度为 `[6,57,4]`、边界输出维度为 `[6,2]`，均已通过有限数值检查。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算短波、长波通量、加热率和边界短波通量的 RMSE，并绘制垂直剖面与误差对照。所有指标均为有限数值，生成的对比图已通过格式和有效像素检查。评估结果保存到：
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

本仓库为 WRF-ML 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY 4.0 许可证；论文、WRF、RRTMG、ONNX Runtime 及相关数据仍应按照各自许可证及使用条款使用。
