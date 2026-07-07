# PDENNEval

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 PDENNEval 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/PDENNEval" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

PDENNEval 是面向偏微分方程求解任务的 OneScience CFD 模型集合，保留了 FNO、DeepONet、UNet、UNO、PINO、MPNN、PINN 和 WAN 等方法的训练入口与配置示例。该模型包适合用于 PDEBench 风格 HDF5 数据上的训练、评测和方法对比，输入通常是时空网格场或 PDE 系数场，输出为预测的物理场或训练得到的 checkpoint。

本仓库整理为 ModelScope 可上传的 OneScience 标准运行包。根目录 `conf/` 下提供经过当前数据集 `OneScience/pdenneval` 预检的标准入口配置：`fno_2d_darcy.yaml` 和 `fno_1d_burgers.yaml`。原始方法子目录中的配置文件完整保留，作为扩展运行参考；网页端大模型默认应优先使用根目录 `conf/`、`scripts/preflight_check.py` 和 `onescience_run_manifest.yaml`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, pde, neural_operator, benchmark |
| 任务 | pde_surrogate_training_evaluation |
| 任务标签 | train, evaluate, preflight, hdf5 |
| 主平台资源 | https://modelscope.cn/models/OneScience/PDENNEval |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/PDENNEval` |
| 必需模型文件 | `FNO/train.py`, `conf/fno_2d_darcy.yaml`, `conf/fno_1d_burgers.yaml`, `scripts/preflight_check.py` |
| 必需数据集 | `OneScience/pdenneval` |
| 支持能力 | 预检、训练、评测；推理需先提供或训练得到 checkpoint |
| 最小验证 | `python scripts/preflight_check.py --all` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文件 | 人类与大模型的第一入口 | 是 | 全部能力 | 模型包根目录 | 正文为中文 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行协议，声明资源身份、关系、命令和输出 | 是 | 全部能力 | 模型包根目录 | 大模型必须解析 |
| `onescience_relations.yaml` | 关系索引 | 模型和数据集的双向关系索引 | 是 | 资源发现 | 模型包根目录 | 与 Manifest relations 一致 |
| `conf/fno_2d_darcy.yaml` | 标准配置 | FNO + 2D Darcy Flow 默认训练/评测配置 | 是 | 预检、训练、评测 | `conf/fno_2d_darcy.yaml` | 已适配 `OneScience/pdenneval` |
| `conf/fno_1d_burgers.yaml` | 标准配置 | FNO + 1D Burgers 默认训练/评测配置 | 是 | 预检、训练、评测 | `conf/fno_1d_burgers.yaml` | 已适配 `OneScience/pdenneval` |
| `scripts/preflight_check.py` | 预检脚本 | 检查配置、数据环境变量、HDF5 文件结构和输出目录 | 是 | 预检 | `scripts/preflight_check.py` | 不加载完整大文件 |
| `FNO/train.py` | 训练脚本 | FNO 训练和评测入口 | 是 | 训练、评测 | `FNO/train.py` | 从 `FNO/` 目录执行 |
| `DeepONet/`, `UNet/`, `UNO/`, `PINO/`, `MPNN/`, `PINN/`, `WAN/` | 原始方法代码 | 保留 PDENNEval 的其他方法代码与配置参考 | 否 | 扩展训练、研究复现 | 对应子目录 | 默认 run_matrix 不直接引用 |
| `run_pdenneval.ipynb` | Notebook | 原始示例 notebook | 否 | 参考 | 模型包根目录 | 手动运行 |

## Manifest

Manifest 文件路径为 `onescience_run_manifest.yaml`。当配置文件、默认数据集、命令或输出目录发生变化时，必须同步更新该 Manifest 中的 `files`、`relations`、`run_matrix`、`commands` 和 `configuration_adaptation` 字段。大模型在自动运行前应先解析 YAML，确认 `resource.id` 与 `platform_resource.primary.repo_id` 都是 `OneScience/PDENNEval`。

## 模型 vs 数据集关系

本模型包必须配套数据集 `OneScience/pdenneval`。模型 Manifest 中通过 `relations.required_datasets` 声明该数据集，数据集 Manifest 中通过 `relations.compatible_models` 反向声明本模型。默认运行场景只引用数据集中已验证可用的 `2D_DarcyFlow_beta0.1_Train.hdf5` 和 `1D_Burgers_Sols_Nu0.001.hdf5`。

## 文件与下载

下载模型包：

```bash
modelscope download --model OneScience/PDENNEval
```

下载数据集：

```bash
modelscope download --dataset OneScience/pdenneval
```

如果使用 `modelscope download --cache_dir`，下载后请先切换到实际模型包根目录，再执行 Manifest 中的命令。例如：

```bash
cd /path/to/downloaded/OneScience/PDENNEval
```

## 环境安装

本包默认运行在已有 OneScience CFD 环境中。缺少 OneScience 时可参考官方仓库安装 CFD 域依赖：

```bash
bash install.sh cfd
```

运行前通常需要 Python、PyTorch、h5py、PyYAML、ruamel.yaml、numpy，以及训练脚本涉及的 OneScience CFD 模块。MPNN、PINN 等扩展方法还可能需要 `torch_geometric`、`torch_cluster` 或 `deepxde`。

## 运行流程

1. 下载模型包和数据集包。
2. 设置数据目录环境变量，指向数据集仓库中的 `data` 目录：

```bash
export ONESCIENCE_PDENNEVAL_DATA_DIR=/path/to/OneScience/pdenneval/data
```

3. 在模型包根目录执行预检：

```bash
python scripts/preflight_check.py --all
```

4. 训练 FNO + 2D Darcy Flow：

```bash
cd FNO
python train.py ../conf/fno_2d_darcy.yaml
```

5. 训练 FNO + 1D Burgers：

```bash
cd FNO
python train.py ../conf/fno_1d_burgers.yaml
```

如需评测已有 checkpoint，请在相应 YAML 中设置 `training.if_training: False` 和 `training.model_path`，再执行同一个 `train.py` 入口。

## 预检与诊断

预检脚本会检查：

- `ONESCIENCE_PDENNEVAL_DATA_DIR` 是否存在。
- `conf/fno_2d_darcy.yaml` 和 `conf/fno_1d_burgers.yaml` 是否指向标准数据目录。
- 必需 HDF5 文件是否存在，`tensor`、`nu`、坐标数据集的维度与 dtype 是否符合预期。
- 输出目录是否可写。

常见错误：

| 错误现象 | 可能原因 | 处理方式 |
|---|---|---|
| `ONESCIENCE_PDENNEVAL_DATA_DIR is not set` | 未设置数据目录环境变量 | 设置为下载后数据集包的 `data` 目录 |
| `missing HDF5 data file` | 数据集未下载或路径不对 | 运行 `modelscope download --dataset OneScience/pdenneval` 并重新设置环境变量 |
| `datapipe.source.data_dir must be ${ONESCIENCE_PDENNEVAL_DATA_DIR}` | 使用了未适配的原始配置 | 优先使用根目录 `conf/` 下的标准配置 |
| `ModuleNotFoundError` | OneScience 或深度学习依赖缺失 | 安装 OneScience CFD 域依赖和对应 Python 包 |
| CUDA 显存不足 | HDF5 文件和 batch 较大 | 调小配置中的 `batch_size`、`reduced_batch` 或使用更大显存设备 |

## 输出说明

默认训练输出写入：

- `outputs/fno_2d_darcy/`
- `outputs/fno_1d_burgers/`

训练脚本通常会生成 checkpoint、训练日志和验证损失。具体文件名由原始 `FNO/train.py` 控制，Manifest 的 `expected_outputs` 给出默认预期。

## 限制与适用范围

当前标准包只把已与本次数据目录匹配的 FNO Darcy 与 FNO Burgers 场景列为默认自动运行场景。数据集中的 `1D_Advection_Sols_beta1.0.hdf5` 已上传并可读，但原始多数 Advection 配置引用 `beta0.1` 文件，未作为默认网页端运行场景。其他方法子目录保留原始配置，扩展使用前需要按数据文件名、PDE 元数据、依赖和显存重新预检。

## 引用与许可证

PDENNEval 参考论文：`PDENNEval: A Comprehensive Evaluation of Neural Network Methods for Solving PDEs`。原始数据主要来自 PDEBench / DaRUS 以及项目自生成数据。许可证信息以上游 OneScience、PDENNEval 和数据来源为准；上传到 ModelScope 前如需公开分发，请再次确认数据授权。
