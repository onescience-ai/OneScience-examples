---
license: apache-2.0
language:
- en
- zh
tags:
- OneScience
- 流体力学
- 拉格朗日粒子仿真
- 图神经网络
- 长时序物理预测
frameworks: PyTorch
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">LagrangianMGN</span>
  </strong>
</p>

# 模型介绍
LagrangianMGN 是 DeepMind 与 Stanford University 提出的用于复杂物理系统仿真的图网络模拟器。该模型将物理系统表示为由粒子构成的图结构，通过节点和边上的学习式消息传递建模粒子间相互作用，并预测粒子加速度以逐步更新系统状态。它主要适用于流体仿真、颗粒材料仿真、可变形材料建模、刚体与多材料交互、长时序物理预测以及传统粒子物理仿真器加速等场景。


# 仓库说明

本仓库是 OneScience 整理的 LagrangianMGN 标准运行包，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

* 训练
* 推理
* 评估与可视化
* 生成 DeepMind Lagrangian TFRecord 假数据用于流程连通性验证

当前不支持能力：
* 不内置预训练权重
* 不负责下载外部数据库进行适配


## 适用场景

| 场景      |    说明                          |
| ------- | --------------------------- |
| 粒子流体仿真  | 模拟水体、飞溅、液体流动等粒子化流体现象        |
| 可变形材料建模 | 模拟黏塑性材料、软物质等复杂形变过程          |
| 多材料交互仿真 | 处理流体、颗粒、刚体和可变形材料之间的相互作用     |
| 长时序物理预测 | 通过自回归 rollout 预测数百到数千步的系统演化 |


# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | 元信息 | 保持最小配置 |
| `conf/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train.py` | 训练脚本 | 支持单卡和 torchrun 多卡 |
| `scripts/inference.py` | 推理脚本 | 读取 `weight/checkpoints`，输出 rollout 结果和指标 |
| `scripts/result.py` | 评估与可视化脚本 | 读取 `result/inference/rollout_metrics.npz`，生成 `error.png` |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成 `metadata.json` 和 `train/valid/test.tfrecord` |
| `model/meshgraphnet.py` | 模型文件 | MeshGraphNet Encode-Process-Decode 实现 |
| `weight/` | 权重目录 | 默认 checkpoint 保存在 `weight/checkpoints` |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


## 3. 快速开始

### 安装运行环境

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 生成假数据进行流程验证

如需先验证脚本、模型、checkpoint 和结果文件是否能够完整跑通，可使用仓库内置脚本生成最小 DeepMind Lagrangian TFRecord 数据。默认配置指向 `data/fake_lagrangian`：

```bash
python scripts/fake_data.py
```

如需使用真实 DeepMind Lagrangian 数据，可下载 OneScience 社区数据集，并将 `conf/config.yaml` 中的 `data.data_dir` 指向包含 `metadata.json` 和 `<split>.tfrecord` 的目录：

```bash
modelscope download --dataset OneScience/lagrangian --local_dir ./data/
```

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练会在 `weight/checkpoints` 下保存 `checkpoint.*.mdlus` 文件。

### 推理

```bash
python scripts/inference.py
```

推理脚本会从 `resume_dir` 读取 checkpoint，默认路径为 `weight/checkpoints`。如果该目录下没有可用 checkpoint，脚本会打印 warning 并使用随机初始化权重继续跑通流程。

推理结果会保存至 `result/inference/`，包括 `sequence_*.npz` 和 `rollout_metrics.npz`。默认不生成动画；如需保存 GIF，可执行：

```bash
python scripts/inference.py inference.save_animations=true
```

### 评估和可视化

```bash
python scripts/result.py
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证
- LagrangianMGN 原始论文：[Learning to simulate complex physics with graph networks](https://arxiv.org/abs/2002.09405)。
- 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
