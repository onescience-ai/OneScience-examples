
<p align="center">
  <strong>
    <span style="font-size: 30px;">FuXi</span>
  </strong>
</p>

# 模型介绍

FuXi（伏羲）是复旦大学提出的多阶段全球天气预报模型，通过 short → medium → long 三阶段级联训练，以 0.25° 分辨率实现 15 天全球预报。每个阶段输出作为下一阶段的输入，逐级扩展预报时长。

论文：FuXi: A cascade machine learning forecasting system for 15-day global weather forecast

https://arxiv.org/abs/2306.12873

# 仓库说明

本仓库是 OneScience 整理的 Fuxi 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 生成轻量级 ERA5 HDF5 测试数据。
- 三阶段级联训练（short → medium → long）
- 基于各阶段训练权重独立推理
- 计算 RMSE/ACC，绘制推理结果可视化图像

当前不支持能力：

- 不随包提供真实 ERA5 数据或预训练权重。
- 默认配置面向 720 x 1440 输入网格，完整训练需要较高显存和存储。
- 虚拟数据只用于流程连通性验证，不代表模型效果。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 ERA5 HDF5 数据训练 Fuxi（short/medium/long 三阶段） |
| 本地快速验证 | 使用虚拟数据检查数据读取、模型训练与推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train_short.py` | short 阶段训练脚本 | 起始训练入口，支持单卡和 torchrun 多卡 |
| `scripts/train_medium.py` | medium 阶段训练脚本 | 需要 short 权重 + short 推理结果 |
| `scripts/train_long.py` | long 阶段训练脚本 | 需要 medium 权重 + medium 推理结果 |
| `scripts/inference.py` | 推理脚本 | 按阶段执行，如 `python scripts/inference.py short` |
| `scripts/result.py` | 评估与可视化脚本 | 按阶段执行，如 `python scripts/result.py short` |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成各阶段所需空壳数据用于快速连通性验证 |
| `scripts/data_loader.py` | 本地数据加载器 | medium/long 阶段训练使用 |
| `model/fuxi.py`	| 模型文件	| OneScience整合梳理复现的代码库 |
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

### 生成假数据进行流程验证

如需先用空壳数据检查路径和数据格式，可将 `conf/config.yaml` 中 `max_epoch`、`finetune_step` 同时设为 `1`，并确认 `datapipe.dataset.data_dir` 指向本地测试目录。

```bash
python scripts/fake_data.py
```

同时，OneScience 社区提供可供训练的 ERA5 数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认 `conf/config.yaml` 中数据路径设置正确：

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


# 数据格式

（真实数据存放在 `data/` 下，）默认结构如下：

```text
data/
  data/
    1979.h5
    1980.h5
    ...
  static/
    geopotential.nc
    land_sea_mask.nc
```

年度 HDF5 文件需包含：

- `fields` 数据集，形状为 `[T, C, H, W]`
- `fields.attrs["variables"]`，变量名列表
- `fields.attrs["time_step"]`，时间间隔小时数
- `global_means`，形状为 `[1, C, 1, 1]`
- `global_stds`，形状为 `[1, C, 1, 1]`

 `conf/config.yaml` 中的数据配置：

```yaml
data_dir: 存放ERA5年度数据、均值/标准差文件、静态文件
train_time: [2000, 2001]   # 训练年份
val_time: [2002]            # 验证年份
test_time: [2003]           # 测试年份
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Fuxi 原始论文：FuXi: A cascade machine learning forecasting system for 15-day global weather forecast（https://arxiv.org/abs/2306.12873）。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。