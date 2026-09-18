<p align="center"><strong><span style="font-size: 30px;">Delta-HBV</span></strong></p>

# 模型介绍

Delta-HBV 是可微、质量守恒的物理信息水文模型。参数网络驱动 16 个并行 HBV 组件并动态预测 β 和 γ。

论文：The suitability of differentiable, physics-informed machine learning hydrologic models for ungauged regions and climate change impact assessment  
https://doi.org/10.5194/hess-27-2357-2023

# 模型描述

该模型由 Pennsylvania State University 和 KAUST 团队提出。模型使用 CAMELS 日强迫、35 个流域属性和流量观测训练。模型通过可微 HBV 和动态参数学习，适用于无资料流域、无资料区域和历史流量趋势保持评估。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 无资料流域 | 评估空间样本外径流。 |
| 物理变量预测 | 同时输出流量与蒸散发。 |
| 无资料区域 | 评估连续留出区域的空间外推能力。 |
| 历史趋势评估 | 比较年均和高低流量趋势保持能力。 |
| ModelScope/OneCode 运行 | 验证数据、训练、推理、水文指标和可视化。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode
[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装
```bash
modelscope download --model OneScience/Delta-HBV --local_dir ./Delta-HBV
cd Delta-HBV
```

### 环境依赖
**硬件要求**
- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本。

**DCU环境**
```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍
论文使用 CAMELS 的 671 个流域、35 个静态属性和逐日降水、温度、PET，训练实例包含一年 warm-up 和一年评分期。虚拟数据保留 35 属性、730 日序列、365 日 warm-up、16 个 HBV 组件和动态 β/γ，仅减少流域数和训练轮数。结果仅用于工程验证。

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
训练在 warm-up 后通过流量 RMSE 端到端优化参数网络和可微 HBV，单卡与双进程 DDP 均已通过。训练结果保存到：
```text
result/checkpoints/delta_hbv.pt
result/training/metrics.json
```

### 训练权重
本仓库不在 `weight/` 中内置权重。论文未提供可直接加载的官方预训练权重，因此不提供权重链接；作者的模型代码归档不作为训练权重引用。

### 推理
```bash
python scripts/inference.py
```
推理恢复 checkpoint，并输出 12 个流域的 730 日流量和 ET 序列。输出 shape 和有限数值均已验证。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化
```bash
python scripts/result.py
```
评估计算中位 NSE、ET RMSE 和趋势误差，并生成流量过程图。指标与 PNG 均通过有效性检查。评估结果保存到：
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

本仓库为 Delta-HBV 公开规格的独立工程复现版本。

原始论文采用 CC BY 4.0 许可证；原始论文、官方代码以及 CAMELS 和 MODIS 数据仍应按照各自项目的许可证及使用条款使用。
