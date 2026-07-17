---
license: apache-2.0
language:
- en
- zh
tags:
- OneScience
- PDE
- CFD
- neural-operator
- PDEBench
frameworks: PyTorch
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">PDENNEval</span>
  </strong>
</p>

# 模型介绍
PDENNEval 是中山大学团队提出的用于神经网络 PDE 求解方法评测的综合基准系统。它不是单一 PDE 求解模型，而是集成并比较 12 种代表性神经网络方法，包括 PINN、DRM、WAN、DFVM、FNO、DeepONet、PINO、U-NO、MPNN 和 U-Net 等，用于评估不同方法在多类 PDE 问题上的精度、效率、鲁棒性和泛化表现。它主要适用于 PDE 神经网络方法横向评测、神经算子与物理约束方法对比、复杂科学计算任务基准测试，以及流体、材料、金融、电磁等领域的 PDE 求解方法选型。

# 仓库说明

本仓库是 OneScience 整理的 PDENNEval 标准运行包，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

* 训练
* 测试
* 评估可视化
* 多模型定义 smoke 测试

当前不支持能力：

* 不内置预训练权重
* 不自动下载真实 PDEBench/PDENNEval 数据集
* 不默认训练全部 PDENNEval 方法；非 FNO 方法当前以模型定义和 smoke 验证为主

## 适用场景

| 场景 | 说明 |
| --- | --- |
| PDE 神经算子训练验证 | 使用 FNO 在 PDEBench 风格 HDF5 数据上完成最小训练和推理闭环 |
| 神经网络 PDE 方法扩展 | `model/` 已包含 FNO、DeepONet、PINO-FNO、UNO、MPNN、UNet 等模型定义 |
| 标准运行包上传 | 工程包含 `configuration.json`、统一配置、脚本入口和空权重目录 |
| 快速连通性检查 | 使用 `fake_data.py` 和 `smoke_models.py` 验证数据、模型和脚本可运行 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | ModelScope/OneCode 元信息 | 保持最小配置 |
| `conf/config.yaml` | 训练、推理和数据配置 | 默认使用 2D Darcy/FNO 小规模配置 |
| `model/fno.py` | FNO 模型定义 | 默认训练和推理使用 |
| `model/deeponet.py` | DeepONet 模型定义 | 用于扩展和 smoke 验证 |
| `model/pino_fno.py` | PINO 使用的 FNO 模型定义 | 用于扩展和 smoke 验证 |
| `model/uno.py` | UNO 模型定义 | 依赖 OneScience 的 `integral_operators` 工具 |
| `model/mpnn.py` | MPNN 模型定义 | 依赖 PyTorch Geometric；PDE 类型仅用于类型检查 |
| `model/unet.py` | UNet 模型定义 | 支持 1D/2D/3D UNet |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成 2D Darcy HDF5：`tensor`、`nu` 和坐标数据集 |
| `scripts/preflight_check.py` | 数据和输出目录预检脚本 | 检查默认 HDF5 schema 和写入权限 |
| `scripts/train.py` | FNO 训练脚本 | 默认输出到 `weight/` |
| `scripts/inference.py` | FNO 推理脚本 | 默认读取 `weight/best_model.pt` |
| `scripts/result.py` | 推理指标汇总脚本 | 统计 `.npz` 推理结果的 MSE/MAE |
| `scripts/smoke_models.py` | 多模型 smoke 脚本 | 覆盖 DeepONet、FNO、MPNN、PINO-FNO、UNet、UNO |
| `weight/` | 权重目录 | 默认仅保留 `.gitkeep`，不随包分发权重 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/PDENNEval --local_dir ./PDENNEval
cd PDENNEval
```

### 安装运行环境

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 生成假数据进行流程验证

默认配置使用 `conf/config.yaml` 中的 `datapipe.source.data_dir: ./data` 和 `file_name: 2D_DarcyFlow_beta0.1_Train.hdf5`。可先生成一个小型 2D Darcy HDF5 文件验证流程：

```bash
python scripts/fake_data.py --overwrite
```

生成后可执行预检：

```bash
python scripts/preflight_check.py
```

### 训练

```bash
python scripts/train.py
```

训练默认运行 1 个 epoch，并在 `weight/` 下写入：

* `latest_model.pt`
* `best_model.pt`
* `model_epoch_0.pt`

如需使用真实 PDENNEval/PDEBench 数据，可下载 OneScience 社区数据集，并将 `conf/config.yaml` 中的 `data.data_dir` 指向包含 `metadata.json` 和 `<split>.tfrecord` 的目录：

```bash
modelscope download --dataset OneScience/pdenneval  --local_dir ./data
python scripts/train.py --data-dir /path/to/data --output-dir ./weight
```

### 推理

```bash
python scripts/inference.py
```

推理默认读取 `weight/best_model.pt`，并将 `.npz` 结果写入 `result/output/`。也可以手动指定 checkpoint：

```bash
python scripts/inference.py --checkpoint ./weight/best_model.pt
```

### 评估

```bash
python scripts/result.py
```

脚本会读取 `result/output/*.npz`，计算 MSE/MAE，并写入 `result/metrics.json`。

### 多模型 smoke 验证

```bash
python scripts/smoke_models.py
```

该脚本会用随机小张量分别实例化并前向运行 DeepONet、FNO、MPNN、PINO-FNO、UNet 和 UNO。若环境中的 `torch_geometric` 可导入但可选 CUDA 扩展缺失，可能会出现 `torch-scatter` 或 `torch-cluster` 的 warning；只要脚本输出 `[OK]`，说明模型定义 smoke 验证通过。


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- PDENNEval 原始论文：[PDENNEval: A Comprehensive Evaluation of Neural Network Methods for Solving PDEs](https://www.ijcai.org/proceedings/2024/0573.pdf)。
- 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
