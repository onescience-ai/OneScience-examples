<p align="center"><strong><span style="font-size: 30px;">AI-GAMFS</span></strong></p>

# 模型介绍

AI-GAMFS 是机器学习驱动的全球气溶胶与气象预报系统。模型以三小时时间间隔生成五天气溶胶和气象预报。

论文：Advancing operational global aerosol forecasting with machine learning  
https://doi.org/10.1038/s41586-026-10234-y

# 模型描述

该模型由中国气象科学研究院、国家气象中心、NASA 等机构团队提出。模型使用 1980–2021 年 MERRA-2 的 54 个气溶胶和气象变量训练。模型通过 Vision Transformer、U-Net 和 3/6/9/12 小时 relay，适用于全球 AOD、气溶胶组分和空气质量预报。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球气溶胶预报 | 预测 AOD、组分光学厚度和地表浓度。 |
| 沙尘与烟霾 | 跟踪区域污染输送事件。 |
| 气象耦合 | 联合预测气溶胶与气象状态。 |
| ModelScope/OneCode 运行 | 验证数据、训练、推理、气溶胶指标和可视化。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/AI-GAMFS --local_dir ./AI-GAMFS
cd AI-GAMFS
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
论文使用 1980–2023 年 MERRA-2 的 12 个气溶胶变量、6 个地表变量和 4 个九层高空变量，共 54 个通道。虚拟数据保留 54 通道、全球逻辑网格和四种 relay 时效，仅减少 tile、宽度和轮数。结果仅用于工程验证。
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
训练分别优化 3、6、9、12 小时模型，单卡与双进程 DDP 均已通过。训练结果保存到：
```text
result/checkpoints/ai_gamfs.pt
result/training/metrics.json
```

### 训练权重
本仓库不在 `weight/` 中内置权重。论文未给出可直接加载的官方预训练权重链接，因此不提供权重链接。

### 推理
```bash
python scripts/inference.py
```
推理恢复 checkpoint 并生成 40 个三小时时效的 54 通道预报。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化
```bash
python scripts/result.py
```
评估计算 AOD RMSE 和相关系数并生成时效曲线，指标与 PNG 均通过有效性检查。评估结果保存到：
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
本仓库为 AI-GAMFS 公开规格的独立工程复现版本。

原始论文采用 CC BY-NC-ND 4.0 许可证；原始论文、官方代码、模型权重和相关数据仍应按照各自项目的许可证及使用条款使用。
