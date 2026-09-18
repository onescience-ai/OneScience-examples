<p align="center"><strong><span style="font-size: 30px;">FuXi-Weather</span></strong></p>

# 模型介绍

FuXi-Weather 是从原始卫星观测到全球天气预报的机器学习系统。系统联合 FuXi-DA 分析和 FuXi Short/Medium 级联预报。

论文：A data-to-forecast machine learning system for global weather  
https://doi.org/10.1038/s41467-025-62024-1

# 模型描述

该系统由上海科学智能研究院、复旦大学、中国气象局等团队提出。模型使用 ERA5、三颗极轨卫星微波亮温和 GNSS-RO 数据训练。模型通过掩码观测潜空间同化和级联预报，适用于六小时循环分析与十天全球天气预报。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 卫星数据同化 | 融合稀疏观测和背景场。 |
| 全球预报 | 级联执行短期和中期预报。 |
| 循环分析 | 每六小时更新全球分析场并启动新预报。 |
| ModelScope/OneCode 运行 | 验证结构化数据、训练、推理、天气指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode
[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装
```bash
modelscope download --model OneScience/FuXi-Weather --local_dir ./FuXi-Weather
cd FuXi-Weather
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
论文使用 0.25° ERA5、极轨卫星微波亮温和 GNSS-RO，并以 8 小时观测窗执行六小时循环同化。虚拟数据保留观测、掩码、背景场、全球逻辑 shape 和级联协议，仅减少 tile、通道工程账本和 rollout。结果仅用于工程验证，不代表论文性能。

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
训练实际联合优化分析和预报目标，单卡与双进程 DDP 均已通过。训练结果保存到：
```text
result/checkpoints/fuxi_weather.pt
result/training/metrics.json
```

### 训练权重
本仓库不在 `weight/` 中内置权重。论文作者公开的 FuXi 模型位于 https://zenodo.org/records/10401602 ，本文使用的 FuXi Weather 模型位于 https://zenodo.org/records/15762985 。

### 推理
```bash
python scripts/inference.py
```
推理恢复 checkpoint，并在 Short/Medium 切换点执行 12 个工程时效。输出 `[2,12,20,16,16]` 已通过 shape 和有限数值检查。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化
```bash
python scripts/result.py
```
评估计算逐时效 RMSE 并生成误差曲线，指标与 PNG 均通过有效性检查。评估结果保存到：
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

本仓库为 FuXi-Weather 公开规格的独立工程复现版本。

原始论文采用 CC BY-NC-ND 4.0 许可证；原始论文、官方代码、模型权重和相关数据仍应按照各自项目的许可证及使用条款使用。
