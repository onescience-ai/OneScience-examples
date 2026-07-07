# DeepCFD

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 DeepCFD 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/DeepCFD" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

DeepCFD 是一个面向稳态层流管道绕障问题的卷积神经网络代理模型，输入几何和符号距离场，输出速度分量与压力场。本仓库是 OneScience 标准运行包，包含训练入口、推理入口、适配后的配置文件、运行前预检脚本和机器可读 Manifest。

模型默认配套 ModelScope 数据集 `OneScience/deepcfd` 使用。数据集提供 `dataX.pkl` 和 `dataY.pkl`，每个文件包含 981 个样本、3 个通道、172x79 网格，默认按 `split_ratio=0.7` 划分为 686 个训练样本和 295 个测试样本。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, laminar_flow, convolutional_surrogate |
| 任务 | steady_laminar_flow_surrogate |
| 任务标签 | train, inference, evaluation, visualization |
| 主平台资源 | https://modelscope.cn/models/OneScience/DeepCFD |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/DeepCFD` |
| 必需模型文件 | `train.py`, `inference.py`, `conf/deepcfd.yaml`, `scripts/preflight_check.py` |
| 必需数据集 | `OneScience/deepcfd` |
| 支持能力 | 预检、训练、分布式训练、推理、评测、可视化 |
| 最小验证 | `ONESCIENCE_DEEPCFD_DATA_DIR=/path/to/OneScience_deepcfd/data python scripts/preflight_check.py --skip-checkpoint` |

能力和命令要求

| 能力 | 必须提供 |
|---|---|
| `preflight` | `python scripts/preflight_check.py --skip-checkpoint`，检查配置、数据路径、PKL shape/dtype、切分和输出目录 |
| `train` | `python train.py`，使用 `conf/deepcfd.yaml` 和 `OneScience/deepcfd` 数据 |
| `finetune` | 暂未声明 |
| `inference` | `python inference.py`，需要 `result/deepcfd/best_model.pt` |
| `evaluate` | `python inference.py` 输出测试 batch 误差与可视化 |
| `visualize` | `python inference.py` 输出 `result/deepcfd/vis_results/` |
| `deploy` | 暂未声明 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 文档 | 人类用户和大模型入口，说明文件、关系、下载、预检和运行方式 | 是 | 全部能力 | 模型包根目录 | 中文正文 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源 ID、数据集关系、命令、输出和诊断 | 是 | 全部能力 | 模型包根目录 | 大模型必须解析 |
| `conf/deepcfd.yaml` | 配置文件 | DeepCFD 数据管道、模型和训练配置 | 是 | 预检、训练、推理、评测 | `conf/deepcfd.yaml` | 已将数据路径适配为 `${ONESCIENCE_DEEPCFD_DATA_DIR}` |
| `scripts/preflight_check.py` | 预检脚本 | 检查配置、数据目录、PKL 文件、shape、dtype、切分和输出目录 | 是 | 预检 | `scripts/preflight_check.py` | 不启动训练 |
| `train.py` | 训练脚本 | 构建 DeepCFDDatapipe 和 UNetEx，训练并保存 checkpoint | 是 | 训练 | `train.py` | 输出到 `result/deepcfd/` |
| `inference.py` | 推理脚本 | 加载 `best_model.pt`，执行测试 batch 推理和可视化 | 是 | 推理、评测、可视化 | `inference.py` | 没有 checkpoint 时会退出 |
| `slurm.sh` | 集群脚本 | Slurm 训练示例 | 否 | 分布式训练 | `slurm.sh` | 需按集群环境调整 |
| `run_deepcfd.ipynb` | Notebook | 原始 DeepCFD notebook 示例 | 否 | 示例 | `run_deepcfd.ipynb` | 供交互使用 |

## Manifest

机器可读 Manifest 位于仓库根目录 `onescience_run_manifest.yaml`。修改模型入口、配置路径、下载命令、数据集 ID、运行矩阵或输出路径后，必须同步更新该文件，并建议执行 YAML 解析和 `command_refs` 校验。

## 模型 vs 数据集关系

本模型必须配套数据集 `OneScience/deepcfd` 使用。模型 Manifest 的 `relations.required_datasets` 已声明完整 `resource_ref`，指向 `https://modelscope.cn/datasets/OneScience/deepcfd`、`README.md` 和 `onescience_run_manifest.yaml`。数据集 Manifest 也通过 `relations.compatible_models` 反向声明适配模型 `OneScience/DeepCFD`。

默认运行场景使用数据集仓库 `data/` 下的 `dataX.pkl` 和 `dataY.pkl`。整理时实测两个 PKL 文件的数组 shape 均为 `(981, 3, 172, 79)`、dtype 为 `float32`；标准包按真实数据 schema 进行预检。

## 文件与下载

下载模型：

```bash
modelscope download --model OneScience/DeepCFD
```

下载数据集：

```bash
modelscope download --dataset OneScience/deepcfd
```

如果使用 `--cache_dir` 下载，请先 `cd` 到实际下载后的模型包根目录再执行运行命令。数据集下载后，将环境变量指向数据集仓库中的 `data` 目录：

```bash
export ONESCIENCE_DEEPCFD_DATA_DIR=/path/to/OneScience_deepcfd/data
```

## 环境安装

```bash
bash install.sh cfd
```

还需要运行环境中可导入 `onescience`、`torch`、`numpy`、`matplotlib`、`tqdm`、`ruamel.yaml`。

## 运行流程

### 1. 环境预检

```bash
python - <<'PY'
import torch, numpy, matplotlib
import onescience
print("environment ok")
PY
```

### 2. 下载

```bash
modelscope download --model OneScience/DeepCFD
modelscope download --dataset OneScience/deepcfd
```

### 3. 应用运行包和准备文件

```bash
cd /path/to/downloaded/OneScience_DeepCFD
export ONESCIENCE_DEEPCFD_DATA_DIR=/path/to/downloaded/OneScience_deepcfd/data
```

### 4. 运行前预检

```bash
python scripts/preflight_check.py --skip-checkpoint
```

成功时应看到 `[OK] model preflight completed`。如果要在推理前确认 checkpoint，也可以去掉 `--skip-checkpoint`，脚本会检查 `result/deepcfd/best_model.pt` 是否存在。

### 5. 运行

训练：

```bash
python train.py
```

多卡训练：

```bash
mpirun -np <num_GPUs> --allow-run-as-root python train.py
```

推理、评测和可视化：

```bash
python inference.py
```

### 6. 验证输出

训练会在 `result/deepcfd/best_model.pt` 保存最佳 checkpoint。推理会读取测试集首个 batch，在 `result/deepcfd/vis_results/` 下生成可视化图片。

## 输出说明

| 输出路径 | 说明 | 成功标准 |
|---|---|---|
| `result/deepcfd/best_model.pt` | 训练产生的最佳模型权重 | 文件存在且可由 `inference.py` 加载 |
| `result/deepcfd/vis_results/` | 推理可视化图片目录 | `python inference.py` 完成后生成图片 |

## 预检与诊断

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `ONESCIENCE_DEEPCFD_DATA_DIR is not set` | 未设置数据集目录环境变量 | 设置为 `OneScience/deepcfd` 数据集仓库的 `data` 目录 |
| `Data files not found` 或 `missing required data file` | 环境变量指向错误或数据未下载完整 | 确认存在 `dataX.pkl` 和 `dataY.pkl` |
| `shape mismatch` 或 `dtype mismatch` | 数据文件被替换或损坏 | 运行数据集仓库的 `python scripts/validate_deepcfd_dataset.py --full-hash` |
| `ModuleNotFoundError` | OneScience CFD 环境未安装或未激活 | 执行 `bash install.sh cfd` 并确认 `PYTHONPATH`/环境正确 |
| `Checkpoint not found` | 尚未训练或未放入权重 | 先运行 `python train.py`，或将兼容 checkpoint 放到 `result/deepcfd/best_model.pt` |

