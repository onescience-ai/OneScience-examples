<p align="center">
  <strong>
    <span style="font-size: 30px;">Pangu-Weather</span>
  </strong>
</p>

# 模型介绍

Pangu-Weather 是华为云提出的全球中期天气预报模型，基于 3D Earth-Specific Transformer 架构，可对地表变量和多气压层高空变量进行快速预测。

论文：Accurate medium-range global weather forecasting with 3D neural networks  
https://www.nature.com/articles/s41586-023-06185-3

# 仓库说明

本仓库是 OneScience 整理的 Pangu-Weather 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 训练
- 推理
- 评估与可视化
- 生成空壳 HDF5 假数据用于流程连通性验证

当前不支持能力：

- 不内置预训练权重
- 不内置真实 ERA5 数据下载与预处理

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 ERA5 HDF5 数据训练 Pangu-Weather |
| 单步推理 | 加载训练得到的 `data/checkpoints/model_bak.pth` 生成预测结果 |
| 结果评估 | 对推理输出计算 RMSE/ACC 并绘制示例变量 |
| 流程验证 | 使用 `fake_data.py` 生成小样本空壳数据检查脚本连通性 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train.py` | 训练脚本 | 支持单卡和 torchrun 多卡 |
| `scripts/inference.py` | 推理脚本 | 需存在训练权重 |
| `scripts/result.py` | 评估与可视化脚本 | 读取 `result/output/*.npy` |
| `scripts/fake_data.py` | 假数据生成脚本 | 用于快速连通性验证 |
| `model/pangu.py/` | 模型文件 | OneScience复现的经典TOP模型 |
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

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 生成假数据进行流程验证

如需先用空壳数据检查路径和数据格式，可将 `config/config.yaml` 中 `model.max_epoch` 设为 `1`，并确认 `datapipe.dataset.data_dir` 指向本地测试目录。

```bash
python scripts/fake_data.py
```

同时，OneScience社区提供可供训练的ERA5数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认'config/config.yaml'中数据路径设置正确；

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

真实数据存放在 `data/` 下，默认结构如下：

```text
data/
  data/
    1979.h5
    1980.h5
    ...
  static/
    land_mask.npy
    soil_type.npy
    topography.npy
```

年度 HDF5 文件需包含：

- `fields` 数据集，形状为 `[T, C, H, W]`
- `fields.attrs["variables"]`，变量名列表
- `fields.attrs["time_step"]`，时间间隔小时数
- `global_means`，形状为 `[1, C, 1, 1]`
- `global_stds`，形状为 `[1, C, 1, 1]`

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Pangu-Weather 原始论文：Accurate medium-range global weather forecasting with 3D neural networks。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。
