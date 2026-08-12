<p align="center">
  <strong>
    <span style="font-size: 30px;">GenCast</span>
  </strong>
</p>

# 模型介绍

GenCast 是基于图神经网络和扩散生成方法的概率性全球天气预报模型。项目使用 JAX/Haiku 实现 GenCast 去噪损失和 DPM-Solver++ 集合采样流程，并通过规则经纬网格与多分辨率网格之间的信息传递生成多成员中期天气预报。

论文：GenCast: Diffusion-based ensemble forecasting for medium-range weather

https://arxiv.org/abs/2312.15796

# 仓库说明

本仓库提供 GenCast 的独立训练、推理和结果可视化流程。`model/data_loader.py` 将 ERA5 HDF5 数据转换为模型所需的具名 `xarray.Dataset` 协议，`model/gencast.py` 封装模型配置、统计量、训练损失和采样接口。

当前支持能力：

- 生成轻量 ERA5 HDF5 测试数据、静态场和 GenCast 按层统计量文件。
- 使用单设备 JAX `jit` 或同一主机多设备 JAX `pmap` 训练。
- 从本项目训练 checkpoint 或配置的官方 GenCast checkpoint 执行集合自回归推理。
- 将每个集合成员和 lead time 流式保存为 NetCDF，或保存为单个集合 NetCDF 文件。
- 绘制指定变量的集合均值和集合离散度图。

当前不支持能力：

- 仓库不内置真实 ERA5 数据、官方统计量或预训练权重。
- 默认配置使用 `mesh_size: 4` 的 GenCast Mini 结构；完整 `mesh_size: 6` 配置需要显著更多计算资源。
- 数据适配器只接受项目定义的官方 WB13 变量、气压层和 6/12 小时时间间隔协议。
- 虚拟数据只用于检查数据格式和流程连通性，不代表模型效果。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 概率天气预报训练 | 使用符合 GenCast WB13 协议的 ERA5 HDF5 数据训练扩散模型。 |
| 集合自回归推理 | 使用训练 checkpoint 或官方 checkpoint 生成多成员、多个 12 小时预报步。 |
| 本地快速验证 | 使用小网格虚拟数据检查数据适配、训练、推理和可视化入口。 |
| 多卡训练 | 通过 JAX `pmap` 在同一主机的多个本地设备上进行数据并行训练。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明 | 中文为主 |
| `conf/config.yaml` | 运行平台、数据、模型、采样、训练和输出配置 | 默认 GenCast Mini、4 个集合成员、30 个预报步 |
| `scripts/fake_data.py` | 生成测试 ERA5、静态场和统计量 | 默认生成 9 x 16 小网格 |
| `scripts/train.py` | JAX 训练入口 | 支持 `single` 和 `pmap` 模式 |
| `scripts/inference.py` | 集合自回归推理入口 | 支持训练 checkpoint 和官方 checkpoint |
| `scripts/result.py` | 集合预报可视化入口 | 绘制集合均值和离散度 |
| `model/common.py` | 配置、统计量和训练 checkpoint 工具 | checkpoint 格式为 `.npz` |
| `model/data_loader.py` | ERA5 HDF5 到 GenCast xarray 协议的适配器 | 校验 WB13 变量、气压层和网格 |
| `model/gencast.py` | GenCast JAX/Haiku 模型封装 | 构造损失与采样变换 |
| `model/graphcast/` | GenCast/GraphCast 基础组件 | 包含网格、稀疏 Transformer、扩散采样和 rollout 实现 |
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

## 3. 快速开始

### 生成假数据进行流程验证

若无真实数据，可先生成 `conf/config.yaml` 中训练和测试年份对应的虚拟 ERA5 数据、静态场及统计量。默认小网格为 9 x 16，并满足 `width = 2 x (height - 1)` 的网格约束。

```bash
python scripts/fake_data.py
```


同时， OneScience 社区提供可供训练的 ERA5 数据。下载后请确认 `conf/config.yaml` 中 `data.data_dir`、`data.static_dir` 和 `data.stats_dir` 指向实际目录。

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
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --config conf/config.yaml --parallel-mode pmap --num-devices 2 --global-batch-size 2
# CUDA_VISIBLE_DEVICES 指定使用的设备索引。
# --num-devices 指定使用的设备数量。
# --global-batch-size 必须能被设备数量整除。
```

训练后权重保存于 `data/checkpoints/model_bak.npz`。

### 训练权重

推理默认读取 `data/checkpoints/model_bak.npz`。如需使用官方 checkpoint，可在 `conf/config.yaml` 中设置 `inference.official_checkpoint`，或通过 `--checkpoint` 传入文件路径。仓库当前未内置预训练权重。

### 推理

```bash
python scripts/inference.py
```

默认配置生成 4 个集合成员和 30 个 12 小时预报步。

### 评估和可视化

```bash
python scripts/result.py
```

脚本默认读取推理结果中的 `2m_temperature`，绘制指定 lead time 的集合均值和集合离散度，并保存为 `result/gencast_forecast.png`。

# 数据格式

默认数据目录结构如下：

```text
data/
  data/
    2000.h5
    2001.h5
    2003.h5
  static/
    geopotential_at_surface.npy
    land_mask.npy
  stats/
    mean_by_level.nc
    stddev_by_level.nc
    diffs_stddev_by_level.nc
    min_by_level.nc
  checkpoints/
    model_bak.npz
```

年度 HDF5 文件需包含：

- `fields` 数据集，形状为 `[T, C, H, W]`。
- `fields.attrs["variables"]`，需包含 6 个地表变量，以及温度、位势、U/V 风、垂直速度和比湿在 13 个气压层上的变量。
- `fields.attrs["time_step"]`，只能为 6 或 12 小时，并需与 `precipitation_interval_hours` 一致。
- 原始 `total_precipitation` 表示每个源时间间隔的累计量；数据适配器将其组合为 `total_precipitation_12hr`。

静态场要求：

- `geopotential_at_surface.npy`，形状为 `[H, W]`。
- `land_mask.npy`，形状为 `[H, W]`。

统计量要求：

- `mean_by_level.nc`
- `stddev_by_level.nc`
- `diffs_stddev_by_level.nc`
- `min_by_level.nc`

数据适配器会将纬度调整为严格递增，并校验规则经纬网格满足 `W = 2 x (H - 1)`。模型目标协议包含 84 个输出通道。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- GenCast 原始论文：GenCast: Diffusion-based ensemble forecasting for medium-range weather（https://arxiv.org/abs/2312.15796）。
- 本仓库包含基于 Apache License 2.0 的 GenCast/GraphCast 相关实现。
