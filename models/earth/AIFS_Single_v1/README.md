<p align="center">
  <strong>
    <span style="font-size: 30px;">AIFS_Single_v1</span>
  </strong>
</p>

# 模型介绍

AIFS (Artificial Intelligence Forecasting System) 是 ECMWF 发布的基于图神经网络（GNN）的全球中期天气预报模型。通过 Encoder-Processor-Decoder 架构在 N320 高斯网格（542,080 节点）与 o96 简化网格之间进行消息传递，使用 16 层滑动窗口 Transformer 实现全球大气状态的 10 天预报。

论文：*AIFS — ECMWF's data-driven forecasting system*, arXiv:2406.01465

https://arxiv.org/abs/2406.01465

# 仓库说明

本仓库是 OneScience 整理的 AIFS Single v1.1 最小可运行独立模型仓库，面向本地快速验证和流程连通性测试场景。

当前支持能力：

- 生成轻量级 ERA5 HDF5 测试数据和N320网格坐标数据。
- AIFS_v1.1单卡训练
- 使用训练权重推理并保存 `.npy` 预测结果。
- 计算 RMSE/ACC，并绘制基于变量的预测结果图。

当前不支持的能力：

- 不随包提供真实 ERA5 数据或预训练权重。
- 默认配置面向 720 x 1440 输入网格，完整训练需要较高显存和存储。
- 虚拟数据只用于流程连通性验证，不代表模型效果。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 ERA5 HDF5 数据从零训练 AIFS |
| 本地快速验证 | 使用虚拟数据检查数据读取、模型训练与推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |


# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train.py` | 训练脚本 | 支持从零训练 |
| `scripts/inference.py` | 推理脚本 | 自回归多步预报，需训练权重 |
| `scripts/result.py` | 评估与可视化脚本 | 读取推理输出 |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成测试用 ERA5 H5 数据 |
| `model/aifs.py` | 模型封装 | 基于onescience的模型复现 |
| `model/aifs_config.json` | 架构配置 | 从官方 checkpoint 提取 |
| `model/grid-n320.npz` | N320 网格坐标 | 从官方 checkpoint 提取 |
| `weights/` | 权重目录 | 权重存放位置 |
| `download.sh` | 下载AIFS网格数据 | 一键下载脚本 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本。

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

如需先用空壳数据检查路径和数据格式，可将 `conf/config.yaml` 中 `training.max_steps` 设为 `5`，以快速验证训练流程
```bash
python scripts/fake_data.py
```
同时，OneScience 社区提供可供训练的 ERA5 数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认 `conf/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

### 下载网格数据
```bash
bash download.sh
```
执行后，网格数据文件（.npz）会下载到 model 目录下

### 训练

```bash
python scripts/train.py
```

训练权重保存至 `weights/model_bak.ckpt`，训练前计算得到的归一化文件保存至`weights/era5_stats.npz`


### 推理

```bash
python scripts/inference.py
```

预报步数通过 `conf/config.yaml` 中 `test_lead_time` 控制（小时数，默认 24 = 1 天）。
推理结果将保存至 `output`目录


### 评估和可视化

```bash
python scripts/result.py
```

计算 ACC / RMSE 指标并绘图。指标保存至 `metrics/`，图片保存至 `plots/`。


# 数据格式

真实数据存放在 `data/` 下，默认结构如下：

```text
data/
  data/
    2005.h5
    2006.h5
    ...
```

HDF5 文件包含 `fields` 数据集，形状为 `[T, C, H, W]`，附带 `fields.attrs["variables"]` 变量名列表和 `time_step` 属性。

`conf/config.yaml` 中的数据配置：

```yaml
data:
  data_dir: "./fake_era5"
  train_years: [2005, 2006]
  val_years:   [2007]
  test_years:  [2008]
  test_lead_time: 24  # hours (240 = 10 days)
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- AIFS 原始论文：Lang et al., *AIFS — ECMWF's data-driven forecasting system*, arXiv:2406.01465。
- 本仓库基于 anemoi-models (0.5.0) 和 anemoi-graphs 构建，遵循 Apache License 2.0。
