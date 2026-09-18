<p align="center">
  <strong>
    <span style="font-size: 30px;">Pangu-Weather</span>
  </strong>
</p>

# 模型介绍

Pangu-Weather 是华为云提出的全球中期天气预报模型，可对地表变量和多气压层高空变量进行快速预测。

论文：Accurate medium-range global weather forecasting with 3D neural networks  
https://www.nature.com/articles/s41586-023-06185-3

# 模型描述

Pangu-weather基于 3D Earth-Specific Transformer 架构，使用ERA5数据进行训练，面向短中期开展天气预报

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 ERA5 HDF5 数据训练 Pangu-Weather |
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
modelscope download --model OneScience/Pangu-Weather --local_dir ./pangu_weather
cd pangu_weather
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

OneScience社区提供可供训练的ERA5数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确；

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
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

训练会在 `data/checkpoints/` 下保存 `model_bak.pth`。

### 训练权重

本仓库在`weights/`文件夹内提供基于1979～2025年ERA5数据训练的权重，该权重在6小时预报超过官方开源onnx权重

### 推理

```bash
python scripts/inference.py
```

推理结果会保存至 `result/output/`。

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

- 本仓库为Pangu-Weather原始论文的复现版本。



