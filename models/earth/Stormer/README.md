<p align="center">
  <strong>
    <span style="font-size: 30px;">Stormer</span>
  </strong>
</p>

# 模型介绍

Stormer 是面向中期天气预报的 Transformer 模型，通过视觉 Transformer 主干学习全球气象场随预报时间间隔变化的状态差分。项目实现基于逐变量天气嵌入、二维正弦位置编码和带时间间隔条件的 Transformer 模块，默认使用 1.40625°（128 x 256）网格。

论文：Scaling transformer neural networks for skillful and reliable medium-range weather forecasting

https://arxiv.org/abs/2312.03876

# 仓库说明

本仓库是 Stormer 的独立 PyTorch 运行项目，配置文件位于 `conf/config.yaml`，训练、推理和评估由 `scripts/` 下的脚本提供。

当前支持能力：

- 生成符合项目数据读取流程的 ERA5 HDF5 测试数据、静态场和 Stormer 归一化文件。
- 使用 6、12、24 小时训练间隔进行单卡训练或 `torchrun` 多卡训练。
- 使用训练权重对 6 小时和 72 小时 lead time 执行自回归推理，并保存 `.npy` 结果。
- 计算各变量 RMSE/ACC，绘制训练/验证损失曲线和样例预报图。

当前不支持能力：

- 仓库不内置真实 ERA5 数据或预训练权重。
- 默认 128 x 256 网格、24 层 Transformer 和 1024 隐藏维度对显存要求较高。
- 虚拟数据只用于检查数据格式和流程连通性，不代表模型效果。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 中期天气预报训练 | 使用配置中的 ERA5 HDF5 数据训练 Stormer 差分预测模型。 |
| 本地快速验证 | 使用 `scripts/fake_data.py` 检查数据生成、训练、推理和评估入口。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装 OneScience 地球环境并运行脚本。 |
| 多卡训练 | 使用 `torchrun` 启动 PyTorch DistributedDataParallel 训练。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明 | 中文为主 |
| `conf/config.yaml` | 模型、数据、归一化和分布式配置 | 默认 128 x 256、69 个输入变量 |
| `scripts/fake_data.py` | 生成测试 HDF5、静态场和 `.npz` 归一化文件 | 输出到配置中的数据和归一化目录 |
| `scripts/train.py` | 训练入口 | 保存 `model_bak.pth`、`trloss.npy` 和 `valoss.npy` |
| `scripts/inference.py` | 自回归推理入口 | 输出到 `result/output/` |
| `scripts/result.py` | 评估和可视化入口 | 输出 RMSE/ACC 和图片到 `result/` |
| `model/stormer.py` | Stormer 模型实现 | 输出归一化状态差分 |
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

`scripts/fake_data.py` 会根据 `conf/config.yaml` 生成 HDF5 年度文件、`static/` 静态场和 `normalize/` 下的输入/差分归一化文件。虚拟数据只用于流程连通性验证。

```bash
python scripts/fake_data.py
```

同时，OneScience 社区提供可供训练的 ERA5 数据。下载后请确认 `conf/config.yaml` 中 `datapipe.dataset.data_dir` 指向实际数据目录，并确认 `model.normalize_dir` 下存在 Stormer 归一化文件。

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

训练和推理默认使用 `data/checkpoints/model_bak.pth`。仓库当前未内置预训练权重。

### 推理

```bash
python scripts/inference.py
```

推理默认生成结果到 `result/output/`。

### 评估和可视化

```bash
python scripts/result.py
```


# 数据格式

默认数据目录结构如下：

```text
data/
  data/
    2000.h5
    2001.h5
    2002.h5
    2003.h5
  static/
    lat.npy
    lon.npy
    geopotential.nc
    land_sea_mask.nc
  normalize/
    normalize_mean.npz
    normalize_std.npz
    normalize_diff_mean_6.npz
    normalize_diff_std_6.npz
    normalize_diff_mean_12.npz
    normalize_diff_std_12.npz
    normalize_diff_mean_24.npz
    normalize_diff_std_24.npz
```

年度 HDF5 文件需包含：

- `fields` 数据集，形状为 `[T, C, H, W]`，其中默认 `H=128`、`W=256`。
- `fields.attrs["variables"]`，变量名列表；默认配置使用 4 个地表变量和 5 组、每组 13 个气压层变量，共 69 个变量。
- `fields.attrs["time_step"]`，默认配置为 6 小时。
- `global_means` 和 `global_stds`，用于保持 ERA5 HDF5 数据协议完整；Stormer 的输入和状态差分归一化由 `normalize/` 下的 `.npz` 文件提供。

`normalize/` 中的 `.npz` 文件使用变量名作为键；输入归一化文件和 6/12/24 小时差分归一化文件都必须包含 `conf/config.yaml` 中配置的变量。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Stormer 原始论文：Scaling transformer neural networks for skillful and reliable medium-range weather forecasting（https://arxiv.org/abs/2312.03876）。
- 本仓库为 Stormer 原始论文的复现版本。
