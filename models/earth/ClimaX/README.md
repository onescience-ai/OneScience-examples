<p align="center">
  <strong>
    <span style="font-size: 30px;">ClimaX</span>
  </strong>
</p>

# 模型介绍

ClimaX 是首个为天气和气候科学设计的通用基础模型，由微软研究院与加州大学洛杉矶分校的研究人员联合开发，相关论文发表于 ICML 2023。

论文：ClimaX: A foundation model for weather and climate

https://arxiv.org/abs/2301.10343


# 模型描述

ClimaX 基于 Vision Transformer 架构实现，首次将多源气候数据预训练与下游任务（区域预报、气候投影等）统一到同一框架中。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 ERA5 HDF5 数据训练 ClimaX |
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
modelscope download --model OneScience/ClimaX --local_dir ./ClimaX
cd ClimaX
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

OneScience社区提供可供训练的ERA5数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确：

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
torchrun --nproc_per_node=8 scripts/train.py
```

训练会在 `data/checkpoints/` 下保存 `model_bak.pth`。

### 训练权重

本仓库在`weights/`文件夹内提供基于ERA5数据训练的权重，权重文件即将上传，预计将于近期完成。

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

- 本仓库为 ClimaX 原始论文的复现版本（MIT License）。


