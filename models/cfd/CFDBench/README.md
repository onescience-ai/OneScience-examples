# CFDBench

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 CFDBench 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/CFDBench" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

CFDBench 是面向计算流体动力学替代建模的大规模基准，覆盖顶盖驱动方腔流、管道流、坝流和圆柱绕流。每个问题包含边界条件、物性参数和几何参数变化子集，输入通常是速度场、case 参数和时间步，输出是后续或目标时刻的二维速度场。

本仓库是 OneScience 标准运行包，包含自回归入口 `train_auto.py`、非自回归入口 `train.py`、适配后的配置、预检脚本、Slurm 示例和机器可读 Manifest。默认场景使用 `OneScience/cfdbench` 数据集的 `tube_prop_bc_geo` 子集，以 `fno` 自回归模型执行训练、测试、评测和可视化。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, fluid_dynamics, benchmark, surrogate_model |
| 任务 | cfd_surrogate_benchmark |
| 任务标签 | train, inference, evaluation, autoregressive, static |
| 主平台资源 | https://modelscope.cn/models/OneScience/CFDBench |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/CFDBench` |
| 必需模型文件 | `train_auto.py`, `train.py`, `conf/cfdbench.yaml`, `scripts/preflight_check.py` |
| 必需数据集 | `OneScience/cfdbench` |
| 支持能力 | 预检、训练、分布式训练、推理、评测、可视化 |
| 最小验证 | `ONESCIENCE_CFDBENCH_DATA_DIR=/path/to/OneScience_cfdbench/data python scripts/preflight_check.py` |

能力和命令要求

| 能力 | 必须提供 |
|---|---|
| `inference` | `python train_auto.py`，配置中 `mode=train_test` 时训练后自动测试 |
| `train` | `python train_auto.py` 或 `python train.py` |
| `finetune` | 暂未声明 |
| `evaluate` | `python train_auto.py` 生成 `scores.json` |
| `visualize` | `python train_auto.py` 生成测试图片 |
| `deploy` | 暂未声明 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 文档 | 人类用户和大模型入口，说明文件、关系、下载、预检和运行方式 | 是 | 全部能力 | 模型包根目录 | 中文正文 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源 ID、数据集关系、命令、输出和诊断 | 是 | 全部能力 | 模型包根目录 | 大模型必须解析 |
| `conf/cfdbench.yaml` | 配置文件 | CFDBench 数据、模型和训练配置 | 是 | 预检、训练、推理、评测 | `conf/cfdbench.yaml` | 已适配为 `${ONESCIENCE_CFDBENCH_DATA_DIR}`、`tube_prop_bc_geo`、`auto`、`fno` |
| `scripts/preflight_check.py` | 预检脚本 | 检查配置、数据路径、case 文件、schema、数组可读性和输出目录 | 是 | 预检 | `scripts/preflight_check.py` | 不启动训练 |
| `train_auto.py` | 运行脚本 | 自回归模型训练、测试、评测和可视化入口 | 是 | 训练、推理、评测、可视化 | `train_auto.py` | 默认入口 |
| `train.py` | 运行脚本 | 非自回归 DeepONet/FFN 训练和测试入口 | 是 | 训练、评测 | `train.py` | 切换 static 配置时使用 |
| `slurm.sh` | 集群脚本 | Slurm 训练示例 | 否 | 分布式训练 | `slurm.sh` | 需按集群环境调整 |
| `run_cfdbench.ipynb` | Notebook | CFDBench notebook 示例 | 否 | 示例 | `run_cfdbench.ipynb` | 供交互使用 |

## Manifest

机器可读 Manifest 位于仓库根目录 `onescience_run_manifest.yaml`。修改模型入口、配置路径、下载命令、数据集 ID、运行矩阵或输出路径后，必须同步更新该文件，并建议执行 YAML 解析和 `command_refs` 校验。

## 模型 vs 数据集关系

本模型必须配套数据集 `OneScience/cfdbench` 使用。模型 Manifest 的 `relations.required_datasets` 已声明完整 `resource_ref`，指向 `https://modelscope.cn/datasets/OneScience/cfdbench`、`README.md` 和 `onescience_run_manifest.yaml`。数据集 Manifest 也通过 `relations.compatible_models` 反向声明适配模型 `OneScience/CFDBench`。

默认运行场景读取数据集中的 `data/tube/prop`、`data/tube/bc` 和 `data/tube/geo`。当前配置与数据匹配：三个子集存在，case 目录均包含 `case.json`、`u.npy`、`v.npy`。

## 文件与下载

下载模型：

```bash
modelscope download --model OneScience/CFDBench
```

下载数据集：

```bash
modelscope download --dataset OneScience/cfdbench
```

如果使用 `--cache_dir` 下载，请先 `cd` 到实际下载后的模型包根目录再执行运行命令。数据集下载后，将环境变量指向数据集仓库中的 `data` 目录：

```bash
export ONESCIENCE_CFDBENCH_DATA_DIR=/path/to/OneScience_cfdbench/data
```

## 环境安装

```bash
bash install.sh cfd
```

还需要运行环境中可导入 `onescience`、`torch`、`numpy`、`matplotlib`、`tqdm` 和 `pyyaml`。

## 运行流程

### 1. 环境预检

```bash
python - <<'PY'
import torch, numpy
import onescience
print("environment ok")
PY
```

### 2. 下载

```bash
modelscope download --model OneScience/CFDBench
modelscope download --dataset OneScience/cfdbench
```

### 3. 应用运行包和准备文件

```bash
cd /path/to/downloaded/OneScience_CFDBench
export ONESCIENCE_CFDBENCH_DATA_DIR=/path/to/downloaded/OneScience_cfdbench/data
```

### 4. 运行前预检

```bash
python scripts/preflight_check.py
```

成功时应看到 `[OK] model preflight completed`。

### 5. 运行

自回归 FNO 训练、测试、评测和可视化：

```bash
python train_auto.py
```

多卡训练：

```bash
mpirun -np <num_GPUs> --allow-run-as-root python train_auto.py
```

非自回归模型入口：

```bash
python train.py
```

### 6. 验证输出

训练会在 `result/**/ckpt-*/model.pt` 保存 checkpoint，并写出 `train_losses.json`。测试会输出 `result/**/test/scores.json`、`preds.pt` 和 `images/*.png`。

## 输出说明

| 输出路径 | 说明 | 成功标准 |
|---|---|---|
| `result/**/ckpt-*/model.pt` | 训练产生的模型权重 | 文件存在且可由测试阶段加载 |
| `result/**/train_losses.json` | 训练损失曲线数据 | JSON 可解析 |
| `result/**/test/scores.json` | 测试指标 | 包含 nmse 等指标 |
| `result/**/test/images/*.png` | 可视化图片 | 测试阶段生成 |

## 预检与诊断

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `ONESCIENCE_CFDBENCH_DATA_DIR is not set` | 未设置数据集目录环境变量 | 设置为 `OneScience/cfdbench` 数据集仓库的 `data` 目录 |
| `Problem path not found` 或 `No cases found` | 环境变量指向错误或数据未下载完整 | 确认存在 `data/tube/prop`、`data/tube/bc`、`data/tube/geo` |
| `u/v shape mismatch` | 同一 case 的 u/v 文件版本不一致 | 运行数据集仓库的 `python scripts/validate_cfdbench_dataset.py --full-hash` |
| `ModuleNotFoundError` | OneScience CFD 环境或依赖缺失 | 安装 OneScience CFD 运行环境和 Python 依赖 |
| 显存或内存不足 | CFDBench 数据较大、batch 较大 | 降低 `batch_size` 和 `eval_batch_size`，或先切换到较小子集 |

## 限制与适用范围

本标准包默认配置为 `tube_prop_bc_geo`、`auto`、`fno`。数据集还包含 cavity、dam、cylinder 以及其他子集；切换实验时需要同步修改 `conf/cfdbench.yaml` 的 `data_name`、`task_type`、`model.name`，并同步更新 `onescience_run_manifest.yaml` 的运行矩阵。

## 引用与许可证

参考论文：`CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid Dynamics`，arXiv `https://arxiv.org/abs/2310.05963`。当前整理包未在原始目录中发现明确许可证文件，上传前如需公开分发应补充许可证信息。
