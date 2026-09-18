<p align="center"><strong><span style="font-size: 30px;">RISE-UNet</span></strong></p>

# 模型介绍

RISE-UNet 是融合深度学习和动力模式预报的次季节根区土壤湿度模型。模型递归预测未来五周土壤湿度异常，并以集合方式评估干旱概率。

论文：Skillful subseasonal soil moisture drought forecasts with deep learning-dynamic models  
https://doi.org/10.1038/s41467-025-62761-3

# 模型描述

该模型由 Auburn University 团队提出。模型使用 GLEAM 根区土壤湿度、ERA5 再分析以及 GEFSv12 和 ECMWF S2S 再预报数据训练。模型结合 residual、inception、squeeze-and-excitation 与 UNet++ 并递归利用前一周预测，适用于美国本土、中国和澳大利亚的周尺度根区土壤湿度与闪旱预报。

# 适用场景

| 场景 | 说明 |
|---|---|
| 次季节土壤湿度 | 预测周 1–5 的 RZSM 异常。 |
| 干旱预报 | 识别低于第 20 百分位事件。 |
| 集合预报 | 使用 11 个动力模式成员和推理 dropout。 |
| 混合建模 | 融合再分析与动力模式再预报。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、概率降水指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/RISE-UNet --local_dir ./RISE-UNet
cd RISE-UNet
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

论文使用 `48×96` 的 0.5° 区域网格、11 个集合成员以及周尺度历史和预报变量，目标为 GLEAM 0–100 cm 根区土壤湿度异常。虚拟数据保持网格、成员数、递归五周协议和 RISE 核心模块，仅减少初始化样本、宽度和训练轮数。结果仅用于工程验证，不代表论文性能。

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

默认虚拟数据训练能够完成五周递归优化、深监督损失和集合 spread 约束，单卡与双进程 DDP 流程均已验证通过。训练生成可用于递归推理的单一 checkpoint，并记录 CRPSexp 训练结果。训练结果保存到：
```text
result/checkpoints/rise_unet.pt
result/training/metrics.json
```

### 训练权重

论文代码公开于 https://osf.io/6y4kh/ ，但正文未确认独立官方预训练权重及其许可证，因此不提供权重链接。

### 推理

```bash
python scripts/inference.py
```

推理恢复 checkpoint，保留 dropout 随机性并递归生成 11 个成员的周 1–5 土壤湿度预测。输出维度为 `[11,5,48,96]`，并已通过有限数值检查。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算逐周 ACC、CRPS 和干旱 GSS，并生成第 3 周空间误差图。指标均为有限数值，对比图已通过 PNG 格式和有效像素检查；虚拟结果不代表论文性能。评估结果保存到：
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

本仓库为 RISE-UNet 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY-NC-ND 4.0 许可证；论文以及 GLEAM、ERA5、GEFSv12 和 ECMWF S2S 数据仍应按照各自许可证及使用条款使用。
