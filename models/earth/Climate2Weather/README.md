<p align="center"><strong><span style="font-size: 30px;">Climate2Weather</span></strong></p>

# 模型介绍

Climate2Weather 是用于概率气候降尺度的生成式机器学习框架。模型将粗分辨率气候模拟转换为多变量、高分辨率且时空一致的天气轨迹。

论文：A Generative Framework for Probabilistic, Spatiotemporally Coherent Downscaling of Climate Simulation  
https://doi.org/10.1038/s41612-025-01157-y

# 模型描述

该方法由 University of Tübingen 和 Tübingen AI Center 提出。模型使用 2006–2013 年 COSMO-REA6 高分辨率再分析训练，并用气候模式作为推理条件。模型通过 SDA 后验采样实现四变量 `8×8→128×128`、6小时到1小时降尺度。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 概率降尺度 | 生成多成员高分辨率轨迹。 |
| 时空一致性 | 联合建模变量与时间窗口。 |
| 多变量联合生成 | 同时降尺度风、气温和海平面气压。 |
| 气候影响研究 | 为区域影响模型生成精细天气驱动。 |
| ModelScope/OneCode 运行 | 验证数据、训练、推理、概率指标和可视化。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode
[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装
```bash
modelscope download --model OneScience/Climate2Weather --local_dir ./Climate2Weather
cd Climate2Weather
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
论文使用四变量、`8×8` 六小时气候场和 `128×128` 逐小时 COSMO-REA6 目标。虚拟数据保留四变量、三时刻联合窗口、16 倍空间放大和 6 倍时间细化，仅减少样本、网络宽度和扩散步数。结果仅用于工程验证。

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
训练完成联合时空窗口的去噪分数匹配，单卡与双进程 DDP 均已通过。训练结果保存到：
```text
result/checkpoints/climate2weather.pt
result/training/metrics.json
```

### 训练权重
本仓库不在 `weight/` 中内置权重。论文作者在 https://github.com/schmidtjonathan/Climate2Weather 提供扩散模型训练权重与实验代码；本紧凑实现不声明与官方权重兼容。

### 推理
```bash
python scripts/inference.py
```
推理恢复 checkpoint，并生成 8 条联合时空轨迹。输出 `[8,3,4,128,128]`、正集合 spread 和有限数值均已验证。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化
```bash
python scripts/result.py
```
评估计算 RMSE、spread、PIT 和时间差异，并生成空间误差图。指标与 PNG 均通过有效性检查。评估结果保存到：
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

本仓库为 Climate2Weather 公开规格的独立工程复现版本。

原始论文采用 CC BY 4.0 许可证；原始论文、官方代码、模型权重和相关数据仍应按照各自项目的许可证及使用条款使用。
