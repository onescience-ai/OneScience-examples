# BENO

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 BENO 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/BENO" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

BENO 是 Boundary-Embedded Neural Operator，即边界嵌入神经算子。该模型面向 CFD 和偏微分方程代理建模任务，用异构图神经网络把复杂边界形状、非均匀边界值和椭圆 PDE 源项共同编码，用于预测二维网格上的解场。

本仓库是 OneScience 标准运行包，包含训练入口、推理与可视化入口、适配后的配置文件、运行前预检脚本和机器可读 Manifest。运行包默认使用 ModelScope 数据集 `OneScience/beno` 中 `data/Dirichlet` 下的 `N32_4c` 子集，训练 900 个样本、测试 100 个样本。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, elliptic_pde, neural_operator |
| 任务 | elliptic_pde_surrogate |
| 任务标签 | train, inference, evaluation, visualization |
| 主平台资源 | https://modelscope.cn/models/OneScience/BENO |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/BENO` |
| 必需模型文件 | `train.py`, `inference.py`, `conf/beno.yaml`, `scripts/preflight_check.py` |
| 必需数据集 | `OneScience/beno` |
| 支持能力 | 预检、训练、分布式训练、推理、评测、可视化 |
| 最小验证 | `ONESCIENCE_BENO_DATA_DIR=/path/to/OneScience_beno/data python scripts/preflight_check.py` |

能力和命令要求

| 能力 | 必须提供 |
|---|---|
| `inference` | `python inference.py`，需要 `OneScience/beno` 数据和可选 `model/model_epoch_*.pt` |
| `train` | `python train.py`，使用 `conf/beno.yaml` 和 `data/Dirichlet/*N32_4c*` |
| `finetune` | 暂未声明 |
| `evaluate` | `python inference.py` 输出 L2/MAE |
| `visualize` | `python inference.py` 输出 `picture/forcing_solution_comparison.png` |
| `deploy` | 暂未声明 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 文档 | 人类用户和大模型入口，说明文件、关系、下载、预检和运行方式 | 是 | 全部能力 | 模型包根目录 | 中文正文 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源 ID、数据集关系、命令、输出和诊断 | 是 | 全部能力 | 模型包根目录 | 大模型必须解析 |
| `conf/beno.yaml` | 配置文件 | BENO 数据管道、模型和训练配置 | 是 | 预检、训练、推理、评测 | `conf/beno.yaml` | 已将数据路径适配为 `${ONESCIENCE_BENO_DATA_DIR}/Dirichlet/` |
| `scripts/preflight_check.py` | 预检脚本 | 检查配置、数据路径、必需 NPY、shape、dtype、样本数和缓存目录 | 是 | 预检 | `scripts/preflight_check.py` | 不启动训练 |
| `train.py` | 训练脚本 | 构建 BENODatapipe 和 HeteroGNS，训练并保存 checkpoint | 是 | 训练 | `train.py` | 输出到 `model/` |
| `inference.py` | 推理脚本 | 加载最新 checkpoint，执行推理、评测和可视化 | 是 | 推理、评测、可视化 | `inference.py` | 无 checkpoint 时会以随机权重运行 |
| `slurm.sh` | 集群脚本 | Slurm 训练示例 | 否 | 分布式训练 | `slurm.sh` | 需按集群环境调整 |
| `run_beno.ipynb` | Notebook | 原始 BENO notebook 示例 | 否 | 示例 | `run_beno.ipynb` | 供交互使用 |

## Manifest

机器可读 Manifest 位于仓库根目录 `onescience_run_manifest.yaml`。修改模型入口、配置路径、下载命令、数据集 ID、运行矩阵或输出路径后，必须同步更新该文件，并建议执行 YAML 解析和 `command_refs` 校验。

## 模型 vs 数据集关系

本模型必须配套数据集 `OneScience/beno` 使用。模型 Manifest 的 `relations.required_datasets` 已声明完整 `resource_ref`，指向 `https://modelscope.cn/datasets/OneScience/beno`、`README.md` 和 `onescience_run_manifest.yaml`。数据集 Manifest 也通过 `relations.compatible_models` 反向声明适配模型 `OneScience/BENO`。

默认运行场景使用数据集中的 `data/Dirichlet/RHS_N32_4c_all.npy`、`data/Dirichlet/SOL_N32_4c_all.npy` 和 `data/Dirichlet/BC_N32_4c_all.npy`。当前配置与该数据匹配：每个文件 1000 个样本，训练 900 个、测试 100 个，分辨率为 32。

## 文件与下载

下载模型：

```bash
modelscope download --model OneScience/BENO
```

下载数据集：

```bash
modelscope download --dataset OneScience/beno
```

如果使用 `--cache_dir` 下载，请先 `cd` 到实际下载后的模型包根目录再执行运行命令。数据集下载后，将环境变量指向数据集仓库中的 `data` 目录：

```bash
export ONESCIENCE_BENO_DATA_DIR=/path/to/OneScience_beno/data
```

## 环境安装

```bash
bash install.sh cfd
```

还需要运行环境中可导入 `onescience`、`torch`、`torch_geometric`、`torchvision`、`numpy`、`matplotlib`、`tqdm` 和 `omegaconf`。

## 运行流程

### 1. 环境预检

```bash
python - <<'PY'
import torch, numpy, omegaconf
import onescience
print("environment ok")
PY
```

### 2. 下载

```bash
modelscope download --model OneScience/BENO
modelscope download --dataset OneScience/beno
```

### 3. 应用运行包和准备文件

```bash
cd /path/to/downloaded/OneScience_BENO
export ONESCIENCE_BENO_DATA_DIR=/path/to/downloaded/OneScience_beno/data
```

### 4. 运行前预检

```bash
python scripts/preflight_check.py
```

成功时应看到 `[OK] model preflight completed`。

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

训练会在 `model/model_epoch_*.pt` 保存 checkpoint，并在 `cache_data/` 生成图数据缓存。推理会输出 L2/MAE 日志，并生成 `picture/forcing_solution_comparison.png`。

## 输出说明

| 输出路径 | 说明 | 成功标准 |
|---|---|---|
| `model/model_epoch_*.pt` | 训练产生的模型权重 | 文件存在且可由 `inference.py` 加载 |
| `cache_data/cached_N32_4c_train_900.pt` | 训练图缓存 | 训练或预处理后生成 |
| `cache_data/cached_N32_4c_test_100.pt` | 测试图缓存 | 训练或推理预处理后生成 |
| `picture/forcing_solution_comparison.png` | 推理可视化图片 | `python inference.py` 完成后生成 |

## 预检与诊断

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `ONESCIENCE_BENO_DATA_DIR is not set` | 未设置数据集目录环境变量 | 设置为 `OneScience/beno` 数据集仓库的 `data` 目录 |
| `Data file not found` | 环境变量指向错误或数据未下载完整 | 确认存在 `data/Dirichlet/RHS_N32_4c_all.npy` 等文件 |
| `shape mismatch` 或 `dtype mismatch` | 数据文件被替换或损坏 | 运行数据集仓库的 `python scripts/validate_beno_dataset.py --full-hash` |
| `ModuleNotFoundError` | OneScience CFD 环境或依赖缺失 | 安装 OneScience CFD 运行环境和 Python 依赖 |
| `No checkpoints found. Running with random weights.` | 尚未训练或未放入 checkpoint | 先执行 `python train.py`，或放入兼容的 `model/model_epoch_*.pt` |

## 限制与适用范围

本标准包默认配置为 `Dirichlet/N32_4c`，分辨率 32，训练 900 个样本、测试 100 个样本。数据集还包含其他前缀和 Neumann 边界数据，但切换实验时需要同步修改 `conf/beno.yaml` 的 `data_dir` 或 `file_prefix`，并同步更新 `onescience_run_manifest.yaml` 中的运行矩阵。

## 引用与许可证

论文与项目链接：OpenReview `https://openreview.net/forum?id=ZZTkLDRmkg`，arXiv `https://arxiv.org/abs/2401.09323`。当前整理包未在原始目录中发现明确许可证文件，上传前如需公开分发应补充许可证信息。
