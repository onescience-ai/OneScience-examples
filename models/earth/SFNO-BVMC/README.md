<p align="center"><strong><span style="font-size: 30px;">SFNO-BVMC</span></strong></p>

# 模型介绍

SFNO-BVMC 是 *Huge ensembles - Part 1* 的集合天气预报工程复现，组合球面傅里叶神经算子、多个训练 checkpoint 和 centered bred vectors。该方法以低成本构造大量具有初值与模型不确定性的全球集合成员。

论文：Huge ensembles - Part 1: Design of ensemble weather forecasts using spherical Fourier neural operators  
https://doi.org/10.5194/gmd-18-5575-2025

# 模型描述

该模型由 Lawrence Berkeley National Laboratory、加州大学伯克利分校、NVIDIA、Indiana University 等机构的研究团队提出。模型使用 ERA5 的 0.25° 全球再分析数据训练，其中 1979–2015 年用于训练、2018 年用于验证、2020 年用于测试。模型适用于全球中期集合天气预报，以及初值不确定性、模型不确定性和极端事件概率的研究。其核心特点是将球面傅里叶神经算子、多个训练 checkpoint 和成对 centered bred vectors 结合起来，高效构造大规模集合。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球集合预报 | 构建初值和模型不确定性集合。 |
| BVMC 研究 | 验证繁殖向量与多 checkpoint 的联合设计。 |
| 概率评估 | 计算集合均值、spread、spread-skill 和 CRPS。 |
| 本地工程验证 | 使用完整通道与全球坐标协议的采样 tile。 |
| OneCode 运行 | 验证数据、训练、推理、评估与可视化。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/SFNO-BVMC --local_dir ./SFNO-BVMC
cd SFNO-BVMC
```

### 训练数据介绍

真实规格为 0.25° 全球网格、77 个输入通道、74 个输出通道和 6 小时时间步长，训练、验证和测试年份分别采用 1979–2015、2018 和 2020。虚拟数据保持 `[77,721,1440]` 到 `[74,721,1440]` 的逻辑维度与全球原坐标，仅减少实际 tile 数量、模型宽度、训练轮数和执行集合成员。虚拟结果只验证工程流程，不代表论文集合预报性能。

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

训练产物保存到：

```text
result/checkpoints/sfno_bvmc.pt
result/training/metrics.json
```

### 训练权重

论文使用训练过程中的 29 个 checkpoint 构造集合；本仓库未确认与论文完全一致的公开官方权重，因此不提供权重链接。

### 推理

```bash
python scripts/inference.py
```

推理从紧凑工程 checkpoint 派生多个 checkpoint 成员，并对每个成员构造正负 centered bred vectors。默认产生 8 个成员和四个 6 小时时效，结果不称为完整全球预报。推理文件保存到：

```text
result/output/ensemble_predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算集合均值 RMSE、集合 spread、spread-skill ratio 和 CRPS，并生成集合均值误差图。输出包含 `is_complete_global=false`，明确结果仅覆盖原坐标 tile。评估文件保存到：

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 SFNO-BVMC 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY 4.0 许可证；论文、官方权重和数据仍应按照各自项目的许可证及使用条款使用。
