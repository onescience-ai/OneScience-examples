# Transolver-Airfoil-Design

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 Transolver-Airfoil-Design 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/Transolver-Airfoil-Design" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

Transolver-Airfoil-Design 是面向二维翼型外流场预测的 CFD 代理模型运行包，整理自 `onescience/examples/cfd/Transolver-Airfoil-Design/`。它使用 Transolver/Transolver++ 等模型，在 AirfRANS 翼型数据上学习非结构网格中的速度、压力和湍流粘度等物理量，可用于训练、推理、评测和结果可视化。

本仓库已经整理为 OneScience ModelScope 标准运行包。模型本身不内置 AirfRANS 大数据文件，而是通过 `relations.required_datasets` 显式关联数据集 `OneScience/airfrans`；运行时需要把 `ONESCIENCE_AIRFRANS_DATA_DIR` 指向数据集仓库中的 `data/Dataset`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, airfoil, surrogate_modeling |
| 任务 | airfoil_design_surrogate |
| 任务标签 | airfoil_design, flow_field_prediction, cfd_surrogate |
| 主平台资源 | https://modelscope.cn/models/OneScience/Transolver-Airfoil-Design |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/Transolver-Airfoil-Design` |
| 支持能力 | 预检、训练、微调、推理、评测、可视化 |
| 必需模型文件 | `train.py`, `inference.py`, `conf/transolver_airfrans.yaml`, `scripts/preflight_check.py` |
| 必需数据集 | `OneScience/airfrans` |
| 最小验证 | `python scripts/preflight_check.py` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令、输出和诊断 | 是 | 全部能力 | `session_workdir/onescience_run_manifest.yaml` | 本任务主 Manifest |
| `manifest.yaml` | Manifest 文件 | 与标准默认路径兼容的 Manifest 副本 | 是 | 全部能力 | `session_workdir/manifest.yaml` | 便于只查找默认文件名的工具读取 |
| `conf/transolver_airfrans.yaml` | 配置文件 | 模型、数据管道、训练、checkpoint 和结果路径配置 | 是 | 预检、训练、推理、评测 | `session_workdir/conf/transolver_airfrans.yaml` | 已将数据路径适配为 `${ONESCIENCE_AIRFRANS_DATA_DIR}` |
| `scripts/preflight_check.py` | 预检脚本 | 检查配置、数据路径、split manifest、VTK 文件、统计目录和 checkpoint 状态 | 是 | 预检、训练、推理 | `session_workdir/scripts/preflight_check.py` | 训练前必须运行 |
| `train.py` | 运行脚本 | 训练 Transolver 并生成 checkpoint 与归一化统计文件 | 是 | 训练、微调 | `session_workdir/train.py` | 支持单卡和分布式启动 |
| `inference.py` | 运行脚本 | 加载 checkpoint，在 full_test 上推理、评测并保存可视化结果 | 是 | 推理、评测、可视化 | `session_workdir/inference.py` | 需要训练后的 checkpoint 和统计文件 |
| `requirements.txt` | 依赖说明 | 列出运行脚本需要的主要 Python 包 | 否 | 环境修复 | `session_workdir/requirements.txt` | PyTorch/PyG 版本需按设备匹配 |
| `dataset/` | 输出目录 | 训练生成 `mean_in.npy`、`std_in.npy`、`mean_out.npy`、`std_out.npy` | 训练后必需 | 推理、评测 | `session_workdir/dataset/` | 当前包不预置统计文件 |
| `checkpoints/transolver_airfrans/` | 输出目录 | 训练生成 `Transolver.pth` | 推理前必需 | 推理、评测 | `session_workdir/checkpoints/transolver_airfrans/` | 当前包不预置 checkpoint |

## Manifest

本仓库的主 Manifest 文件是 `onescience_run_manifest.yaml`，同时提供 `manifest.yaml` 作为标准默认路径兼容文件。修改资源 ID、文件路径、下载命令、运行命令、数据集关系或配置适配说明后，必须同步更新这两个 Manifest，并重新执行 YAML 解析、command_refs 解析和关系双向校验。

## 模型 vs 数据集关系

模型仓库 ID 必须保持为 `OneScience/Transolver-Airfoil-Design`。配套数据集仓库 ID 必须保持为 `OneScience/airfrans`。模型 Manifest 中的 `relations.required_datasets[0].resource_ref.repo_id` 指向 `OneScience/airfrans`，数据集 Manifest 中的 `relations.compatible_models[0].resource_ref.repo_id` 反向指向 `OneScience/Transolver-Airfoil-Design`。

整理时只修改了标准化模型包中的配置，不修改原始模型目录。配置适配如下：

| 配置文件 | YAML 路径 | 原值 | 当前值 | 原因 |
|---|---|---|---|---|
| `conf/transolver_airfrans.yaml` | `datapipe.source.data_dir` | `${ONESCIENCE_DATASETS_DIR}/Transolver-Airfoil-Design/Dataset` | `${ONESCIENCE_AIRFRANS_DATA_DIR}` | 模型和数据集分仓上传，运行时需要显式指向下载后的数据集目录 |
| `conf/transolver_airfrans.yaml` | `datapipe.data.splits` | `task=full`, `train_name=full_train`, `test_name=full_test` | 保持不变 | 当前 `OneScience/airfrans` 包含 full_train 800 个样本和 full_test 200 个样本，配置与数据匹配 |

## 文件与下载

下载模型包：

```bash
modelscope download --model OneScience/Transolver-Airfoil-Design
```

下载配套数据集：

```bash
modelscope download --dataset OneScience/airfrans
```

如果使用 `modelscope download --cache_dir <dir>`，下载完成后请切换到实际下载后的模型包根目录再运行本仓库命令。

## 环境安装

网站环境已经部署 OneScience 时可直接预检。环境缺失时按 CFD 领域安装：

```bash
bash install.sh cfd
```

可按本包依赖文件补充安装：

```bash
pip install -r requirements.txt
```

实际 GPU 训练还需要与本机 CUDA 匹配的 PyTorch 和 PyG 版本。

## 运行流程

### 1. 下载模型和数据集

```bash
modelscope download --model OneScience/Transolver-Airfoil-Design
modelscope download --dataset OneScience/airfrans
```

### 2. 设置数据路径

```bash
export ONESCIENCE_AIRFRANS_DATA_DIR=/path/to/OneScience_airfrans/data/Dataset
```

变量必须指向 `data/Dataset`，不是数据集仓库根目录。

### 3. 运行前预检

```bash
python scripts/preflight_check.py
```

### 4. 训练

```bash
python train.py
```

分布式训练可使用：

```bash
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> train.py
```

### 5. 推理、评测和可视化

```bash
python inference.py
```

推理前需要已有 `checkpoints/transolver_airfrans/Transolver.pth` 和 `dataset/*.npy` 统计文件，通常由训练流程生成。

## 预检与诊断

| 错误现象 | 可能原因 | 处理方式 |
|---|---|---|
| `ONESCIENCE_AIRFRANS_DATA_DIR is not set` | 未设置数据目录环境变量 | 将变量设置为 `OneScience/airfrans` 下载目录中的 `data/Dataset` |
| `manifest.json not found` | 变量指向了错误目录 | 确认变量指向 `data/Dataset`，不是数据集仓库根目录 |
| `Normalization stats not found` | 统计文件还不存在 | 先运行训练生成 `dataset/*.npy`，或放入与 full_train 一致的统计文件 |
| `Checkpoint not found` | 推理缺少 `Transolver.pth` | 先训练生成 checkpoint，或放入兼容 checkpoint |
| `ModuleNotFoundError` | OneScience CFD 依赖未安装完整 | 执行 `bash install.sh cfd` 并检查 `requirements.txt` |

## 输出说明

训练输出位于：

- `checkpoints/transolver_airfrans/Transolver.pth`
- `dataset/mean_in.npy`
- `dataset/std_in.npy`
- `dataset/mean_out.npy`
- `dataset/std_out.npy`

推理和评测输出位于：

- `results/full/score_Transolver.json`
- `results/full/` 下的翼型可视化结果文件

## 限制与适用范围

当前标准包默认使用 AirfRANS `full_train` 和 `full_test` split。仓库不预置训练好的 checkpoint，因此首次推理前需要先完成训练或提供兼容权重。切换到 `scarce`、`reynolds` 或 `aoa` 任务时，需要同步修改配置和 Manifest 中的运行场景说明。

## 引用与许可证

模型代码参考 Transolver 相关开源实现和 AirfRANS 数据集。原始许可证信息需以上游项目为准；本标准包的 `platform_resource.primary.access.license` 暂记为 `unknown`。

本资源遵循 OneScience AI4S ModelScope 大模型运行标准。
