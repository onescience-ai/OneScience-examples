<p align="center">
  <strong><span style="font-size: 30px;">GCRM-ConvectionNN</span></strong>
</p>

# 模型介绍

GCRM-ConvectionNN 用于学习大气柱中的对流加热和增湿倾向，并通过线性响应函数与重力波谱诊断机器学习对流参数化的传播不稳定性，主要用于 GCRM 对流参数化和在线稳定性研究。

论文：Interpreting and Stabilizing Machine-Learning Parametrizations of Convection  
https://doi.org/10.1175/JAS-D-20-0082.1

# 模型描述

该方法由 Vulcan Inc.、University of California Irvine、Columbia University 和 University of Washington 的研究团队提出。论文使用近全球 SAM 云解析模拟粗粒化数据和 SPCAM 超参数化气候模拟数据训练与分析神经网络。模型适用于大气对流参数化、线性稳定性诊断和机器学习物理方案稳定化研究。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| GCRM 对流参数化 | 从 34 层热力学柱状态预测加热和增湿倾向。 |
| 稳定性诊断 | 计算自动微分 LRF、上层输入消融和二维波算子谱。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、谱指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/GCRM-ConvectionNN --local_dir ./GCRM-ConvectionNN
cd GCRM-ConvectionNN
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

训练数据为大气单柱热力学样本，包含 34 层总水混合比和液态冰静力能廓线。输入还包含海表温度和大气顶太阳辐射，共 70 维。目标为同一大气柱的 34 层加热和增湿倾向，共 68 维，时间间隔为 3 小时。本仓库仅使用少量虚拟样本验证训练、稳定性诊断、推理和评估流程，不代表官方数据分布、训练规模或论文正式性能。

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

训练使用质量加权 MSE 和可配置 20 成员平衡正则，检查数据版本与 70/68 维度。训练产物保存到：

```text
result/checkpoints/gcrm_convectionnn.pt
result/training/metrics.json
```

### 训练权重

`weight/` 不包含预训练权重。合成数据产生的工程 checkpoint 不是论文官方 checkpoint。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据 34 层热力学柱状态生成 `Q1/Q2` 对流倾向，并计算完整与上层输入消融后的线性响应和波谱结果。完整数值结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算质量加权 MSE、危险传播模态数、最大增长率和最大传播速度，并保存完整与消融配置的结果；该任务不是多步预测或分类任务，因此不保存逐时效或逐类别指标。评估同时生成 `Q1/Q2` 目标与预测廓线及波谱对比图；虚拟数据结果仅用于工程验证，不代表论文正式性能。

```text
result/evaluation/metrics.json
result/evaluation/profiles_spectrum.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 GCRM-ConvectionNN 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
