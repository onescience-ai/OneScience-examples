<p align="center">
  <strong><span style="font-size: 30px;">AtmoRep</span></strong>
</p>

# 模型介绍

AtmoRep 是基于大规模表征学习的随机大气动力学模型，通过 masked-token 训练和 ensemble 输出学习大气状态分布。

论文：AtmoRep: A stochastic model of atmosphere dynamics using large scale representation learning

https://arxiv.org/abs/2308.13280

# 模型描述

当前目录保留官方 vorticity 单场模型权重、配置和 normalization，并提供 tiny AtmoRep 风格模型用于本地训练和推理验证。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 官方资源校验 | 加载官方 `.mod` 权重并校验配置。 |
| 本地快速验证 | 使用 tiny 模型完成 masked-token 训练和推理。 |
| ERA5 大气表示学习 | 后续可接入官方 GRIB 或 Zarr 数据。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `download.sh` | ModelScope 资源下载脚本 | 下载官方模型权重和 vorticity 归一化文件 |
| `conf/config.yaml` | Tiny 模型训练配置 | 定义样本规模、优化器参数、设备和 checkpoint 路径 |
| `scripts/train.py` | Tiny 模型训练入口 | 执行 masked-token ensemble 训练和验证 |
| `scripts/inference.py` | Tiny 模型推理入口 | 加载本地 checkpoint 并输出 ensemble 预测 |
| `scripts/result.py` | 结果评估入口 | 汇总训练历史与预测统计指标 |
| `scripts/download_official_resources.sh` | 官方资源下载脚本 | 从上游仓库和数据站获取源码及模型归档 |
| `model/tiny_atmorep.py` | Tiny AtmoRep 实现 | 单场 masked-token Transformer 与 ensemble loss |
| `model/fake_data.py` | 合成数据生成模块 | 为本地训练提供确定性的单场测试数据 |
| `resources/` | 官方模型资源目录 | 包含模型配置、权重和 vorticity 归一化文件 |
| `vendor/atmorep-official/` | AtmoRep 官方代码快照 | 用于核对官方配置、模型和数据处理实现 |
| `weight/` | 本地训练权重目录 | 保存 Tiny 模型 checkpoint 和训练历史 |

# 使用说明

## 1. OneCode 使用

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- Tiny 模型可在 CPU 上运行。
- 官方模型和真实数据推理推荐使用 GPU。


## 3. 快速开始
### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

官方路径如缺少依赖，可继续安装：

```bash
pip install zarr wandb cfgrib xarray dask netCDF4 torchinfo
```

### 权重与数据

当前目录保留：

```text
resources/id4nvwbetz/AtmoRep_id4nvwbetz.mod
resources/id4nvwbetz/model_id4nvwbetz.json
resources/data/normalization/vorticity/
```

重新下载官方资源前请注意：当前 vendor 快照不是 Git checkout，现有下载脚本不能向非空 `vendor/atmorep-official` 重新 clone。当前随包资源无需重复下载；脚本仅适合空目标目录。

```bash
bash scripts/download_official_resources.sh .
```

### Tiny 训练

```bash
python scripts/train.py
```

该命令使用独立 train/validation fake Dataset 执行多 epoch masked-token 训练，包含 DataLoader、AdamW、验证、学习率调度、early stopping、最佳/最新 checkpoint 和训练历史。默认参数位于 `conf/config.yaml`。

恢复训练：

```bash
python scripts/train.py --resume weight/training/latest.pth --epochs 20
```

训练产物为 `weight/training/latest.pth`、`best.pth`、`history.json`；同时更新推理兼容权重 `weight/tiny_atmorep.pth`。每个 Dataset 样本为 `fields [T,V,H,W]` 和非空 `mask [N]`，训练/验证使用不同 seed。它是 tiny 模型的完整训练流程，不是论文 35 亿参数官方模型训练。

### Tiny 推理

```bash
python scripts/inference.py
```

推理结果保存为：

```text
result/prediction.pt
result/target.pt
```

其中包括 ensemble、ensemble mean、ensemble std 和 mask。

### 结果检验

```bash
python scripts/result.py
```

该命令生成 `result/metrics.json` 和 `result/comparison.png`。指标为归一化 token 空间的 ensemble/mean/spread RMSE，不是论文的物理量 RMSE、ACC、CRPS 或 spread-skill。

### 论文与当前实现的 I/O

| 项目 | 论文/官方模型 | Tiny smoke 模型 |
| --- | --- | --- |
| 输入 | ERA5 局部 4D 邻域，5 个模式层、多物理场 | `[B,4,1,8,8]` 单场随机张量 |
| token | 变量相关 4D token，含绝对时空与层条件 | `1x4x4` patch，16 token，相对坐标和单层条件 |
| 输出 | 多 head ensemble，支持重建/临近预报/插值 | 4 成员 masked-token ensemble |
| 训练 | 大规模 masked-token 分布学习 | 多 epoch Dataset 训练、独立验证和断点续训 |
| 权重 | `resources/id4nvwbetz` vorticity 官方权重 | `weight/tiny_atmorep.pth`，两者不兼容 |

完整 tiny 执行顺序为 `train.py -> inference.py -> result.py`。结果分析会在存在时读取 `weight/training/history.json`；推理输出包含 `ensemble`、`ensemble_mean`、`ensemble_std`、`mask` 和 `target`。随机数据由 Dataset 按索引生成，当前没有可供官方 Zarr sampler 使用的独立 fake 数据集。模型包发布时不携带本地训练权重或 `result/` 生成物，运行命令后按上述路径创建。

### 官方真实推理

当前尚未提供可直接运行的官方真实推理命令。它还需要 ERA5 vorticity GRIB/Zarr、ecCodes 环境，并需要将官方 `evaluate.py` 中的站点路径参数化。

### 真实数据

官方 vorticity 模型需要 ERA5 vorticity、model levels 96/105/114/123/137、小时级时间轴和 0.25° 全球 GRIB/Zarr 数据。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 官方代码为 MIT License。
- 官方模型权重声明为 CC BY 4.0。
- ERA5 遵循 Copernicus/ECMWF 数据条款。
