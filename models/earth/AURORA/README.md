<p align="center"><strong><span style="font-size: 30px;">AURORA</span></strong></p>

# 模型介绍

AURORA 是微软研究院提出的地球系统基础模型，是一个拥有13亿参数的大型深度学习模型，由三维感知编码器、三维Swin Transformer处理器，动态解码器组成，可完成全球天气预报、空气污染预测等多种地球系统预测任务，相关论文发表于 ICML 2024。

论文：Aurora: A Foundation Model of the Atmosphere

https://arxiv.org/abs/2405.13063


# 模型描述


本仓库是 OneScience 整理的 AURORA 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 生成轻量级 ERA5 HDF5 测试数据
- AURORA 模型的从零构建和训练
- 基于训练权重的独立推理
- 绘制推理结果可视化图像

当前不支持能力：

- 不随包提供真实 ERA5 数据或预训练权重。
- 默认配置面向 720 x 1440 输入网格，完整训练需要较高显存和存储。
- 虚拟数据只用于流程连通性验证，不代表模型效果。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 ERA5 HDF5 数据训练 AURORA |
| 本地快速验证 | 使用虚拟数据检查数据读取，模型训练、微调、推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |


# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train.py` | 训练脚本 | 支持单卡和 torchrun 多卡 |
| `scripts/finetune.py` | 微调脚本 | 支持训练保存权重和官方权重微调 |
| `scripts/inference.py` | 推理脚本 | 需存在训练权重 |
| `scripts/result.py` | 评估与可视化脚本 | 读取 `result/output/*.npy` |
| `scripts/fake_data.py` | 假数据生成脚本 | 用于快速连通性验证 |
| `scripts/validate_data.py` | 数据验证脚本 | 对输入数据进行数据校验 |
| `model/` | 独立 Python 包目录 | 基于官方仓库和 OneScience 实现的 AURORA 源码 |
| `weight/` | 权重目录 | 可放置预训练或发布权重 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

## 3. 快速开始

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```


### 生成虚拟数据进行流程验证

虚拟数据只用于检查数据协议和流程连通性，不代表预报效果：

```bash
python scripts/fake_data.py
```

同时，OneScience 社区提供可供训练的 ERA5 数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认 `conf/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```


### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

以下命令在一台机器上启动 2 个训练进程，每个进程使用一张卡。

```bash
torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```


### 微调


配置已将 `training.finetune.checkpoint` 指向训练阶段生成的 `data/checkpoint/model_bak.pt`，因此默认微调使用训练模型。

```bash
python scripts/finetune.py
```


### 推理

推理默认读取微调保存的 `data/checkpoint/model_finetune.pt`。
```bash
python scripts/inference.py
```


### 评估和可视化

```bash
python scripts/result.py
```


# 数据格式

默认结构如下：

```text
data/
├── data
│   ├── 2000.h5
│   ├── 2001.h5
│   └── ...
├── metadata
│   ├── dataset_card.json
│   ├── generation.json
│   ├── lineage.json
│   ├── splits.json
│   └── statistics.json
└── static
    └── static_vars.npz
```

年度 HDF5 文件需包含：

- `fields` 数据集，形状为 `[T, C, H, W]`
- `fields.attrs["variables"]`，变量名列表
- `fields.attrs["time_step"]`，时间间隔小时数
- `global_means`，形状为 `[1, C, 1, 1]`
- `global_stds`，形状为 `[1, C, 1, 1]`



# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Aurora 论文: https://arxiv.org/abs/2405.13063 。
- 本仓库为 Aurora 原始论文的 OneScience 复现版本，如引用或商用，请电子邮件联系 AIWeatherClimate@microsoft.com，详情请参考官方要求：https://microsoft.github.io/aurora/intro.html 。
- Copyright (c) Microsoft Corporation. Licensed under the MIT license 。
