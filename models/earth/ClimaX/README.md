<p align="center">
  <strong>
    <span style="font-size: 30px;">ClimaX</span>
  </strong>
</p>

# 模型介绍

ClimaX 是面向天气和气候任务的通用基础模型。项目中的模型实现对不同气象变量分别进行 patch tokenization，通过变量嵌入和交叉注意力聚合变量信息，再使用带位置编码和预报时效条件的 Vision Transformer 主干生成目标气象场。

论文：ClimaX: A foundation model for weather and climate

https://arxiv.org/abs/2301.10343

# 仓库说明

本仓库是 ClimaX 的独立 PyTorch 运行项目，默认配置面向 5.625°（32 x 64）ERA5 网格。模型接收 `conf/config.yaml` 中配置的 48 个输入变量，并输出 5 个预报变量。

当前支持能力：

- 生成符合项目数据读取流程的轻量 ERA5 HDF5 测试数据。
- 使用纬度加权 MSE 进行单卡训练或 `torchrun` 多卡训练。
- 按配置中的 `predict_range` 和 `hrs_each_step` 构造 lead time 条件并执行推理。
- 计算输出变量 RMSE/ACC，绘制训练/验证损失曲线和样例预报图。

当前不支持能力：

- 仓库不内置真实 ERA5 数据或预训练权重。
- 当前脚本固定使用配置中的输入变量集合、输出变量集合和空间分辨率，切换数据协议需同步修改配置。
- 虚拟数据只用于检查数据格式和流程连通性，不代表模型效果。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报训练 | 使用 ERA5 HDF5 数据训练 ClimaX 多变量预报模型。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、训练、推理和结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装 OneScience 地球环境并运行脚本。 |
| 多卡训练 | 使用 `torchrun` 启动 PyTorch DistributedDataParallel 训练。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明 | 中文为主 |
| `conf/config.yaml` | 模型、数据和训练配置 | 默认 32 x 64 网格、48 个输入变量、5 个输出变量 |
| `scripts/fake_data.py` | 生成轻量 ERA5 HDF5 测试数据 | 按配置中的年份和变量生成数据 |
| `scripts/train.py` | 训练入口 | 保存 `model_bak.pth`、`trloss.npy` 和 `valoss.npy` |
| `scripts/inference.py` | 推理入口 | 默认读取 `model_bak.pth`，输出到 `result/output/` |
| `scripts/result.py` | 评估和可视化入口 | 输出 RMSE/ACC 和图片到 `result/` |
| `model/ClimaX.py` | ClimaX 模型实现 | 支持按变量选择输出通道 |
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

虚拟数据只用于检查 HDF5 数据格式和脚本连通性。脚本按 `conf/config.yaml` 中的训练、验证和测试年份生成年度文件。

```bash
python scripts/fake_data.py
```


同时，OneScience 社区提供可供训练的 ERA5 数据。下载后请确认 `conf/config.yaml` 中 `datapipe.dataset.data_dir` 指向实际数据目录。

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

推理结果会保存至 `result/output/`。
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
  checkpoints/
    model_bak.pth
    trloss.npy
    valoss.npy
```

年度 HDF5 文件需包含：

- `fields` 数据集，形状为 `[T, C, H, W]`，其中默认 `H=32`、`W=64`。
- `fields.attrs["variables"]`，变量名列表，需包含 `conf/config.yaml` 中 `channels` 配置的全部 48 个输入变量。
- `fields.attrs["time_step"]`，默认配置为 6 小时。
- `global_means` 和 `global_stds`，形状为 `[1, C, 1, 1]`，供推理结果反归一化和评估使用。

默认输出变量为：

- `geopotential_500`
- `temperature_850`
- `2m_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- ClimaX 原始论文：ClimaX: A foundation model for weather and climate（https://arxiv.org/abs/2301.10343）。
- 本仓库为 ClimaX 原始论文的复现版本（MIT License）。
