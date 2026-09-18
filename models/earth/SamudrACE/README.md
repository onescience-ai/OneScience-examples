<p align="center"><strong><span style="font-size: 30px;">SamudrACE</span></strong></p>

# 模型介绍

SamudrACE 是联合三维大气和海洋仿真器的快速耦合气候模型。模型在物理状态空间交换 SST、海冰和表面通量，可执行稳定的长期气候模拟。

论文：SamudrACE: Fast and Accurate Coupled Climate Modeling With 3D Ocean and Atmosphere Emulators  
https://arxiv.org/abs/2509.12490

# 模型描述

该模型由 Ai2、纽约大学、普林斯顿大学、NOAA/GFDL 和哥伦比亚大学等团队提出。模型使用 200 年 GFDL CM4 预工业控制模拟训练。模型通过 ACE2 风格大气组件和 SamudraI 风格海洋组件进行物理状态耦合，适用于海气耦合气候模拟和长期漂移诊断。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 海气耦合模拟 | 每 20 个大气步驱动一个海洋步。 |
| 长期漂移诊断 | 评估大气、海洋热量和盐度代理漂移。 |
| 全球气候仿真 | 保留 1° 全球逻辑网格与多层状态协议。 |
| ModelScope/OneCode 运行 | 验证结构化数据、训练、推理、气候指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/SamudrACE --local_dir ./SamudrACE
cd SamudrACE
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

论文使用共同 1° 全球网格上的 46 个大气通道和 80 个海洋通道，分别保留 8 层与 19 层垂直结构。虚拟数据保持 126 个可枚举通道、全球逻辑 shape 和 20:1 时间耦合，仅减少实际 tile、模型宽度和训练轮数；论文摘要所称 145 场与变量表相差的 19 场不被虚构。结果仅用于工程验证。

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

训练已完成单卡和双进程 DDP 验证，并生成单一可恢复 checkpoint。训练过程中实际执行 20 个大气步和一个海洋步的耦合反向传播，并记录联合 MSE。训练结果保存到：
```text
result/checkpoints/samudrace.pt
result/training/metrics.json
```

### 训练权重

官方权重和初始条件位于 https://huggingface.co/allenai/SamudrACE-CM4-piControl ，本紧凑实现不声明权重兼容。

### 推理

```bash
python scripts/inference.py
```

推理执行四个五天耦合步并保存原坐标 tile 与不完整全球标志。大气和海洋输出均通过 shape 与有限数值检查。恢复后的 checkpoint 保持 46 个大气通道、80 个海洋通道和 20:1 时间耦合协议。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估输出大气和海洋漂移 RMSE、热量与盐度代理序列及 SST 漂移图。指标均为有限数值，PNG 已通过有效性检查。评估结果保存到：
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

本仓库为 SamudrACE 公开规格的独立工程复现版本。

原始论文、官方代码、权重和 GFDL CM4 数据仍应按照各自项目的许可证及使用条款使用。
