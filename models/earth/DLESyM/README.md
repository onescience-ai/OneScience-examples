<p align="center"><strong><span style="font-size: 30px;">DLESyM</span></strong></p>

# 模型介绍

DLESyM 是异步耦合大气与海洋深度学习模块的地球系统模型。模型能够执行长期自由气候模拟并诊断降水。

论文：A Deep Learning Earth System Model for Efficient Simulation of the Observed Climate  
https://arxiv.org/abs/2409.16247

# 模型描述

该模型由相关大气科学与机器学习研究团队提出。模型使用 1983–2017 年 ERA5、ISCCP OLR 和 SST 数据训练。模型通过 DLWP、DLOM 与降水模块异步耦合，适用于当前气候长期模拟和内部变率分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 长期气候模拟 | 执行稳定的大气海洋自由滚动。 |
| 气候变率分析 | 分析 ENSO、季风和环状模态。 |
| 降水诊断 | 从大气状态诊断累计降水。 |
| ModelScope/OneCode 运行 | 验证数据、训练、推理、气候指标和可视化。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/DLESyM --local_dir ./DLESyM
cd DLESyM
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
论文使用 1983–2017 年 ERA5、ISCCP OLR 与 SST，保留 9 个大气预报场和一个海洋 SST 场。虚拟数据保持字段、全球逻辑网格、6 小时大气步、48 小时海洋步和 96 小时耦合周期，仅减少 tile、宽度和 rollout。结果仅用于工程验证。
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
训练联合优化大气、海洋和降水诊断模块，单卡与双进程 DDP 均已通过。训练结果保存到：
```text
result/checkpoints/dlesym.pt
result/training/metrics.json
```

### 训练权重
本仓库不在 `weight/` 中内置权重。作者在 https://github.com/AtmosSci-DLESM/DLESyM 发布完整配置与权重。

### 推理
```bash
python scripts/inference.py
```
推理恢复 checkpoint 并执行四个 96 小时耦合周期，输出大气、SST 和降水场。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化
```bash
python scripts/result.py
```
评估输出大气漂移 RMSE、SST 和降水均值并生成漂移图，指标与 PNG 均通过有效性检查。评估结果保存到：
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
本仓库为 DLESyM 公开规格的独立工程复现版本。

原始预印本、官方代码、模型权重和相关数据仍应按照各自项目的许可证及使用条款使用。
