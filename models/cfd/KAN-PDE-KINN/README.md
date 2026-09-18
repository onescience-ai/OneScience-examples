<p align="center"><strong><span style="font-size: 30px;">KAN-PDE-KINN</span></strong></p>

# 模型介绍

Kolmogorov-Arnold-Informed neural network (arXiv:2406.11045)

论文复现：本模型由 OneScience 基于论文复现，使用 DCU 集群训练，产出可运行代码、配置与 checkpoint。

# 模型描述

模型架构：Kolmogorov-Arnold Network (KAN) with B-spline activations; CPINNs/DEM/BINN variants

主要模型文件：

- `__init__.py`
- `admissible.py`
- `bspline.py`
- `domains.py`
- `evaluate.py`
- `kan.py`
- `losses.py`

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 模型训练 | 使用配置和合成/仿真数据训练模型 |
| 模型推理 | 加载权重进行场预测/生成 |
| 评估与可视化 | 评估指标并生成可视化图 |

# 使用说明

## OneCode 使用

您可以通过 OneCode 平台直接调用本模型进行训练、推理和评估。

## 手动安装使用

### 硬件要求

- GPU / DCU 单卡

### 下载模型包

```bash
modelscope download --model OneScience/KAN-PDE-KINN
```

### 安装运行环境

```bash
conda create -n kan-pde-kinn python=3.11 -y
conda activate kan-pde-kinn
pip install torch numpy pyyaml scipy matplotlib
```

### 训练数据介绍

（请在此处说明训练数据来源和获取方式；复现采用合成/解析数据验证训练闭环）

### 训练

```bash
python scripts/train.py --config conf/crack_binn.yaml
```

### 训练权重

- `kanpde_binn_tier1_model.pt`
- `kanpde_cpinns_tier1_model.pt`
- `kanpde_dem_tier1_model.pt`

### 推理

```bash
python scripts/evaluate.py --config conf/crack_binn.yaml --run-dir <run_dir>
```

### 评估和可视化

```bash
python scripts/evaluate.py --config conf/<config>.yaml --run-dir <run_dir>
python scripts/visualize.py --run-dir <run_dir>
```

# OneScience 官方信息

| 项目 | 地址 |
| --- | --- |
| Gitee | https://gitee.com/onescience |
| GitHub | https://github.com/onescience |

# 引用与许可证

论文引用：Kolmogorov-Arnold-Informed neural network (arXiv:2406.11045)

本模型由 OneScience 复现，遵循 Apache License 2.0。

