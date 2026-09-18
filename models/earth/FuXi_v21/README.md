<p align="center">
  <strong>
    <span style="font-size: 30px;">FuXi_v21</span>
  </strong>
</p>

# 模型介绍

FuXi-2.1（伏羲2.1）是由复旦大学联合上海人工智能实验室（SAIS）开发的一个全球确定性机器学习天气预报模型。其理论根据依然是 FuXi 论文。

论文：FuXi: A cascade machine learning forecasting system for 15-day global weather forecast

https://arxiv.org/abs/2306.12873

# 模型描述

该模型旨在解决AI气象模型预报结果“过于平滑”的行业痛点，在保持均方根误差（RMSE）等传统指标不下降的前提下，生成更清晰、细节更丰富的预报场，从而提升对强降水、大风等极端天气事件的捕捉能力。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报训练 | 使用 C85 ERA5 HDF5 数据训练 FuXi v2.1。 |
| 本地快速验证 | 使用合成 HDF5 数据检查数据读取、训练、推理和推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装依赖并运行脚本。 |
| 多卡训练 | 使用 `torchrun` 运行 DDP 多卡训练。 |

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
modelscope download --model OneScience/FuXi_v21 --local_dir ./FuXi_v21
cd FuXi_v21
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

训练入口使用 OneScience `ERA5Dataset`，数据根目录由 `conf/config.yaml` 的 `paths.data_root` 指定。OneScience 社区提供可供接口验证和训练的数据切片：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

### 生成虚拟数据

真实 HDF5 文件需要包含 `fields` 数据集、C85 变量属性、6 小时时间间隔以及归一化统计量。没有真实数据时，可生成协议兼容的合成文件：

```bash
python scripts/fake_data.py
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

训练检查点默认保存为 `data/checkpoint/model_bak.pth`，指标保存为 `output/training/metrics.json`。

### 训练权重

本仓库在 weights/ 文件夹内提供基于 ERA5 再分析数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理结果默认保存到 `output/inference/forecast.nc`。

### 评估和可视化

```bash
python scripts/result.py
```

默认生成 `figures/fuxi21_t2m.png`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本项目是 FuXi v2.1 的模型复现，不代表复旦大学官方发布的权重或训练配方。
- 本适配目录按 Apache License 2.0 元数据发布；ERA5 数据、OneScience 和 FuXi 上游实现的许可证及使用条件以各自官方声明为准。
