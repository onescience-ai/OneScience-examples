<p align="center"><strong><span style="font-size: 30px;">PrecipExtremes-GAN</span></strong></p>

# 模型介绍

PrecipExtremes-GAN 使用残差生成对抗网络将粗分辨率大气场转换为高分辨率日降水，并评估模型向更暖气候外推极端降水变化的能力。

论文：On the Extrapolation of Generative Adversarial Networks for downscaling precipitation extremes in warmer climates  
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL112492

# 模型描述

该方法由新西兰国家水与大气研究所和新南威尔士大学等机构的研究团队提出。论文使用 ACCESS-CM2 驱动的 CCAM 模拟训练，并在四个独立 GCM 驱动的历史和 SSP3-7.0 模拟上评估。模型适用于新西兰区域的日降水降尺度及极端降水气候变化信号分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 日降水降尺度 | 根据 8 个粗分辨率大气变量生成高分辨率日降水。 |
| 极端降水外推 | 比较历史和未来训练对 99.5 百分位降水变化的影响。 |
| 本地工程验证 | 验证 deterministic U-Net、残差 GAN 和集合推理。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、极端降水指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/PrecipExtremes-GAN --local_dir ./PrecipExtremes-GAN
cd PrecipExtremes-GAN
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

论文使用 1.5° 粗分辨率的 U、V、T 和 Q 在 500/850 hPa 的 8 个通道预测约 12 km CCAM 日降水。公开正文未给最终张量网格点数，本仓库使用 `24×24 → 96×96` 工程网格并明确标记为假设。虚拟数据仅用于验证工程流程，不代表 CCAM 或 GCM 的真实分布、训练规模或论文性能。

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

训练先拟合 deterministic U-Net，再使用 MSE、对抗损失和强度约束训练残差生成器。默认缩小样本、网络宽度、ensemble 和训练轮数，训练产物保存到 `result/checkpoints/precip_extremes_gan.pt` 和 `result/training/metrics.json`。

### 训练权重

论文未提供可直接加载的官方模型权重，本仓库不在 `weight/` 中内置权重。本地 checkpoint 不得描述为官方预训练权重。

### 推理

```bash
python scripts/inference.py
```

推理输出 deterministic baseline、残差 GAN 成员及集合均值，保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估计算日降水 MAE、99.5 百分位误差和未来相对历史气候变化信号误差，并生成对比图。结果保存到 `result/evaluation/metrics.json` 和 `result/evaluation/comparison.png`。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 PrecipExtremes-GAN 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
