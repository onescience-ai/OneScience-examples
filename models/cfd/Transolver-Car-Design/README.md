# Transolver-Car-Design

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 Transolver-Car-Design 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/Transolver-Car-Design/mlcfd_data" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

Transolver-Car-Design 是面向汽车外流场预测的 OneScience CFD 标准运行包。它基于 Transolver/Transolver++ 结构，将车辆几何的非结构化图数据作为输入，预测三维速度分量和表面压力，并进一步评估阻力系数相关误差。

本仓库上传的是可执行模型代码、适配后的配置和预检脚本，不内置预训练 checkpoint。默认模型为 `Transolver_plus`；训练数据、预处理图样本和归一化统计量来自关联数据集 `OneScience/ShapeNetCar`。网页端大模型读取本 README 和 `manifest.yaml` 后，应先下载模型包和数据集包，再执行预检、训练或使用训练产物进行推理评测。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, external_aerodynamics, car_design |
| 任务 | flow_field_prediction |
| 任务标签 | car_design, pressure_prediction, velocity_prediction, drag_coefficient_evaluation |
| 主平台资源 | https://modelscope.cn/models/OneScience/Transolver-Car-Design/mlcfd_data |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/cfd/Transolver-Car-Design` |
| 必需模型文件 | `train.py`, `inference.py`, `conf/transolver_car.yaml`, `scripts/preflight_transolver_car.py` |
| 必需数据集 | `OneScience/ShapeNetCar` |
| 支持能力 | 预检、训练、推理、评测、可视化 |
| 最小验证 | `python scripts/preflight_transolver_car.py --config conf/transolver_car.yaml` |

| 能力 | 必须提供 |
|---|---|
| `inference` | `python inference.py`，需要 `checkpoints/ShapeNetCar/Transolver_plus.pth` 和 `data/mlcfd_data` |
| `train` | `python train.py`，需要 ShapeNetCar 数据和配置文件 |
| `finetune` | 当前未声明 |
| `evaluate` | `python inference.py`，输出误差指标和结果文件 |
| `visualize` | `python inference.py`，当配置中 `save_vtk: True` 且 `visualize: True` 时生成 VTK 和图片 |
| `deploy` | 当前未声明 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和输出 | 是 | 全部能力 | `session_workdir/manifest.yaml` | 修改运行包时必须同步更新 |
| `conf/transolver_car.yaml` | 配置文件 | 模型、数据管道、训练和推理配置 | 是 | 预检、训练、推理、评测、可视化 | `session_workdir/conf/transolver_car.yaml` | 已将数据路径改为 `./data/mlcfd_data/...` |
| `train.py` | 运行脚本 | 从 ShapeNetCar 数据训练 Transolver++ | 是 | 训练 | `session_workdir/train.py` | 输出 checkpoint |
| `inference.py` | 运行脚本 | 加载 checkpoint 推理、评测并可视化 | 是 | 推理、评测、可视化 | `session_workdir/inference.py` | 需要先有 checkpoint |
| `scripts/preflight_transolver_car.py` | 预检脚本 | 检查配置、数据路径、预处理样本 schema 和统计文件 | 是 | 预检 | `session_workdir/scripts/preflight_transolver_car.py` | CPU 即可运行 |
| `data/mlcfd_data/` | 数据目录 | ShapeNetCar 数据集在模型包中的期望位置 | 是 | 预检、训练、推理、评测、可视化 | `session_workdir/data/mlcfd_data/` | 来自 `OneScience/ShapeNetCar` |
| `checkpoints/ShapeNetCar/Transolver_plus.pth` | checkpoint | 训练后生成的权重 | 推理/评测必需 | 推理、评测、可视化 | `session_workdir/checkpoints/ShapeNetCar/Transolver_plus.pth` | 本模型包当前不内置预训练权重 |

## Manifest

完整机器可读运行说明位于仓库根目录 `manifest.yaml`。修改文件路径、下载命令、运行命令、数据集关系或配置适配内容后，必须同步更新该文件，并建议执行 `python ../validate_standardized_repos.py --skip-data-hash` 做结构校验。

## 模型 vs 数据集关系

本模型必须搭配数据集 `OneScience/ShapeNetCar` 使用。模型 Manifest 在 `relations.required_datasets` 中声明了该数据集，并提供完整 `resource_ref`；数据集 Manifest 在 `relations.compatible_models` 中反向声明了本模型。运行场景以 `run_matrix` 为准：最小验证需要 `data/mlcfd_data/training_data`、`data/mlcfd_data/preprocessed_data` 和 `data/mlcfd_data/stats`；训练需要完整 ShapeNetCar 数据；推理和评测还需要训练生成的 checkpoint。

## 文件与下载

下载模型包：

```bash
modelscope download --model OneScience/Transolver-Car-Design/ --local_dir .
```

下载数据集包：

```bash
modelscope download --dataset OneScience/ShapeNetCar --local_dir data
```

如果网页端或脚本使用 `--cache_dir` 下载模型，下载结果可能位于缓存目录下的真实模型包根目录。运行 `python train.py`、`python inference.py` 或预检脚本前，`cwd` 必须切换到包含 `manifest.yaml`、`conf/` 和 `scripts/` 的模型包根目录。

数据集下载后需要保证模型包内存在：

```text
data/mlcfd_data/training_data
data/mlcfd_data/preprocessed_data
data/mlcfd_data/stats
```

## 环境安装

网站环境已部署 OneScience 时不需要重复安装。环境缺失时使用 CFD 领域安装入口：

```bash
bash install.sh cfd
```

## 运行流程

### 1. 下载模型和数据

```bash
modelscope download --model OneScience/Transolver-Car-Design/mlcfd_data --local_dir .
modelscope download --dataset OneScience/ShapeNetCar --local_dir data
```

### 2. 运行前预检

```bash
python scripts/preflight_transolver_car.py --config conf/transolver_car.yaml
```

成功时会输出：

```text
[OK] Transolver-Car-Design preflight passed
```

### 3. 训练

```bash
python train.py
```

训练输出默认写入：

```text
checkpoints/ShapeNetCar/Transolver_plus.pth
```

### 4. 推理、评测和可视化

```bash
python inference.py
```

该命令会读取验证集样本，计算压力、速度和阻力系数相关指标，并在配置允许时生成 VTK 和图片。

### 5. 验证输出

```text
results/ShapeNetCar/Transolver_plus/npy
results/ShapeNetCar/Transolver_plus/vtk
results/ShapeNetCar/Transolver_plus/vis
```

## 预检与诊断

| 错误现象 | 常见原因 | 处理方式 |
|---|---|---|
| `Config file not found` | 当前目录不是模型包根目录 | 切换到包含 `conf/transolver_car.yaml` 的目录 |
| `data root not found` 或 `missing directory` | 没有下载 `OneScience/ShapeNetCar`，或未放到 `data/mlcfd_data` | 执行数据集下载命令并检查目录 |
| `Checkpoint not found` | 尚未训练生成 checkpoint | 先运行 `python train.py`，或提供兼容 checkpoint 到同一路径 |
| `ModuleNotFoundError` | OneScience CFD 依赖未安装或环境未激活 | 执行 `bash install.sh cfd` 或切换到 OneScience 环境 |
| `CUDA out of memory` | 显存不足 | 降低 `datapipe.dataloader.batch_size` 或换更大显存设备 |

## 输出说明

训练输出为 `checkpoints/ShapeNetCar/Transolver_plus.pth`。推理评测输出包括反归一化预测/真值 `.npy` 文件、可视化 VTK 文件和静态图片；日志中会打印压力相对 L2、速度相对 L2、压力 RMSE、速度 RMSE、阻力系数相对误差和 Spearman 相关性。

## 限制与适用范围

本包针对 ShapeNetCar 汽车外流场数据整理，默认模型参数与 `x.npy` 的 7 维输入特征和 `y.npy` 的 4 维输出物理量匹配。当前模型仓库不内置预训练权重，因此推理评测需要先训练或由用户提供兼容 checkpoint。数据集约 18G，默认预检需要完整目录结构。

## 引用与许可证

模型实现参考 Transolver 和 Transolver++ 相关工作以及 OneScience 当前示例工程。数据来源和许可证信息应以 `OneScience/ShapeNetCar` 数据集仓库及原始数据发布方说明为准。
