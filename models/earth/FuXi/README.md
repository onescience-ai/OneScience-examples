<p align="center">
  <strong>
    <span style="font-size: 30px;">FuXi</span>
  </strong>
</p>

# 模型介绍

FuXi（伏羲）是由复旦大学联合多家机构研发的全球天气预报大模型，是首个能够独立完成数据同化（DA）与循环预报的端到端机器学习框架。

论文：FuXi: A cascade machine learning forecasting system for 15-day global weather forecast

https://arxiv.org/abs/2306.12873

# 模型描述

FuXi 模型由 short → medium → long 三阶段级联训练得到，其训练输入主要为 ERA5 再分析数据。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 ERA5 HDF5 数据训练 Fuxi（short/medium/long 三阶段） |
| 本地快速验证 | 使用虚拟数据检查数据读取、模型训练与推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/FuXi --local_dir ./FuXi
cd FuXi
```

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

### 训练数据介绍

OneScience 社区提供可供训练的 ERA5 数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认 `conf/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

### 训练

Fuxi 包含 3 个阶段，**必须按顺序执行**。每个阶段的推理结果作为下一阶段的输入：

**short（训练）→ short（推理）→ medium（训练）→ medium（推理）→ long（训练）→ long（推理）**

**1) 训练 short 模型（从零开始，作为起始训练入口）**

单卡：

```bash
python scripts/train_short.py
```

多卡：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train_short.py
```

**2) 推理 short（生成 medium 的输入数据）**

```bash
python scripts/inference.py short
```

**3) 训练 medium 模型（需要 short 权重 + short 推理结果）**

```bash
python scripts/train_medium.py
```

**4) 推理 medium（生成 long 的输入数据）**

```bash
python scripts/inference.py medium
```

**5) 训练 long 模型（需要 medium 权重 + medium 推理结果）**

```bash
python scripts/train_long.py
```

### 训练权重
本仓库在weights/文件夹内提供基于39年ERA5再分析数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

各阶段可独立推理：

```bash
python scripts/inference.py short
python scripts/inference.py medium
python scripts/inference.py long
```

推理结果会保存至 `result/output/<stage>/`。

### 评估和可视化

```bash
python scripts/result.py short
python scripts/result.py medium
python scripts/result.py long
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 FuXi 原始论文的复现版本。
