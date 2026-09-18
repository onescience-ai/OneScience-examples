<p align="center"><strong><span style="font-size: 30px;">ClimateNet</span></strong></p>

# 模型介绍

ClimateNet 是专家标注的极端天气数据集与像素级分割模型。模型识别热带气旋、大气河和背景区域。

论文：ClimateNet: an expert-labeled open dataset and deep learning architecture for enabling high-precision analyses of extreme weather  
https://doi.org/10.5194/gmd-14-107-2021

# 模型描述

该模型由 LBNL、UC Berkeley、ETH Zurich、NVIDIA、NCAR 等团队提出。模型使用 CAM5.1 的四通道气候场和专家标注掩码训练。模型通过 DeepLabv3+ 语义分割，适用于热带气旋与大气河检测及条件降水分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 极端天气分割 | 识别 TC、AR 和背景像素。 |
| 气候情景分析 | 将模型迁移到不同增暖情景。 |
| 条件降水分析 | 按事件掩码提取降水统计。 |
| ModelScope/OneCode 运行 | 验证数据、训练、推理、分割指标和可视化。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ClimateNet --local_dir ./ClimateNet
cd ClimateNet
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
论文输入为 CAM5.1 的 TMQ、U850、V850 和 PRECT，逻辑 shape 为 `[4,1152,768]`，标签为 BG、TC 和 AR。虚拟数据保持四通道、三分类和原坐标 tile，仅减少样本和执行网格。结果仅用于工程验证。
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
训练使用加权交叉熵并完成单卡和双进程 DDP 验证。训练结果保存到：
```text
result/checkpoints/climatenet.pt
result/training/metrics.json
```

### 训练权重
本仓库不在 `weight/` 中内置权重。作者在 https://portal.nersc.gov/project/ClimateNet/ 提供训练模型与数据。

### 推理
```bash
python scripts/inference.py
```
推理输出 `[4,96,96]` 三类掩码和概率，保留 tile 原坐标与不完整全球标志。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化
```bash
python scripts/result.py
```
评估计算逐类 IoU 和 mean IoU，并生成真值与预测对比图。评估结果保存到：
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
本仓库为 ClimateNet 公开规格的独立工程复现版本。

原始论文采用 CC BY 4.0 许可证；原始论文、官方模型、代码和数据仍应按照各自项目的许可证及使用条款使用。
