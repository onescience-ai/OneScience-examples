<h1 align="center">OpenFold</h1>

## 模型介绍

OpenFold 是 AlphaFold2 的 PyTorch 实现，用于根据蛋白质序列、MSA 和模板信息预测蛋白质三维结构。该模型包含 Evoformer、Structure Module、模板模块、MSA 模块等核心网络结构，可用于训练、推理、权重转换和蛋白质结构预测相关实验。

## 仓库说明

本仓库是 OneScience 整理的 OpenFold 轻量调用仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前目录只抽取并暴露 OpenFold 用户直接需要调用的部分：

- `scripts/`：训练、推理、数据下载、预处理、权重转换等脚本。
- `model/openfold/`：OpenFold 模型层 Python 包。
- `config/`：OpenFold 模型配置、运行配置和 DeepSpeed 配置。
- `data/`：示例数据目录，需下载。
- `weight/`：用户放置预训练、微调或发布权重的位置。
- `configuration.json`：ModelScope/OneCode 元信息。

本仓库不是完整 standalone 版本。数据管线、特征处理、loss、几何工具、np 常量、relax 等公共能力继续依赖 OneScience 基座包提供。使用前需要安装 OneScience，或通过 `ONESCIENCE_ROOT` 指向 OneScience 源码根目录。

当前支持能力：

- 训练：提供 `scripts/train.py`，使用本地 OpenFold 模型层，数据管线等能力来自 OneScience 基座。
- 推理：提供 `scripts/inference.py`，可加载 OpenFold checkpoint 或 AlphaFold JAX 参数进行结构预测。
- 序列 threading：提供 `scripts/thread_sequence.py`，用于自定义模板相关推理场景。
- 工具脚本：`scripts/` 下包含数据库下载、alignment 预计算、mmCIF 缓存生成、权重转换等工具。

当前不支持能力：

- 不脱离 OneScience 基座单独运行完整 OpenFold 数据管线和工具库。
- 不内置完整训练数据集、序列数据库、模板数据库和预训练权重。
- 不保证 CPU 上完成完整训练或大规模推理，CPU 仅建议用于连通性验证。

## 适用场景

| 场景 | 说明 |
| --- | --- |
| 本地 OpenFold 调用 | 用户通过本仓库脚本调用 OpenFold 训练或推理。 |
| OneScience 基座复用 | 继续使用 OneScience 的 datapipes、utils、loss、np 等公共模块。 |
| ModelScope/OneCode 发布 | 只暴露脚本、模型、配置和权重目录，减少发布包体积。 |
| 权重转换和数据准备 | 使用 `scripts/` 下工具下载数据库、预处理 alignment、转换权重。 |

## 文件说明

| 路径 | 功能 | 备注 |
| --- | --- | --- |
| `README.md` | 工程使用说明文档 | 中文为主。 |
| `configuration.json` | ModelScope/OneCode 元信息 | 声明入口脚本、模型包、配置、权重目录和 OneScience 基座依赖。 |
| `config/config.py` | OpenFold 模型配置 | 从 OneScience OpenFold 配置抽取。 |
| `config/config.yaml` | 训练、推理和数据路径配置 | 已适配本仓库相对路径，并声明基座依赖。 |
| `config/deepspeed_config.json` | DeepSpeed 配置 | 用于分布式训练或 ZeRO 相关场景。 |
| `scripts/train.py` | 训练脚本 | 本地模型层 + OneScience 数据管线。 |
| `scripts/inference.py` | 推理脚本 | 需提供 FASTA、模板路径、alignment 或数据库路径，以及权重。 |
| `scripts/thread_sequence.py` | 序列 threading 脚本 | 用于自定义模板相关推理场景。 |
| `scripts/` | 工具脚本目录 | 下载、预处理、权重转换、alignment 等脚本。 |
| `model/openfold/` | OpenFold 模型 Python 包 | 只包含 model 层源码；不包含 OneScience datapipes/utils。 |
| `data/` | 示例数据和用户数据放置目录 | 需下载。 |
| `weight/` | 权重目录 | 可放置 `.pt`、`.ckpt`、`.npz` 等权重文件。 |

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

### 下载data

```bash
modelscope download \
   --model OneScience/OpenFold \
   --include 'data/**' \
   --local_dir ./
```

### 安装运行环境

```bash
git clone https://gitee.com/onescience-ai/onescience.git
cd onescience
bash install.sh bio
```

## 快速验证

```bash
PYTHONPATH=$(pwd)/model  python -c "from openfold.model import AlphaFold; from config.config import model_config; import onescience; print('openfold wrapper ok')"
```

如果提示缺少 `torch`、`ml_collections`、`deepspeed` 等依赖，请先安装 OneScience 推荐运行环境或对应 DCU/GPU 版本依赖。

## 示例数据

`data/` 目录用于放置 OpenFold 运行所需的示例输入和用户自备数据。当前仓库内置 `data/demo/monomer` 单体推理示例，便于快速检查推理脚本、模型导入和 alignment 读取流程。

目录结构：

| 路径 | 说明 |
| --- | --- |
| `data/demo/monomer/fasta_dir/6kwc.fasta` | 示例蛋白序列，FASTA header 为 `6KWC_1`。 |
| `data/demo/monomer/alignments/6KWC_1/` | 与 FASTA header 对应的预计算 MSA/template 搜索结果。 |
| `data/demo/monomer/alignments/6KWC_1/bfd_uniref_hits.a3m` | BFD/UniRef alignment 结果。 |
| `data/demo/monomer/alignments/6KWC_1/uniref90_hits.sto` | UniRef90 alignment 结果。 |
| `data/demo/monomer/alignments/6KWC_1/mgnify_hits.sto` | MGnify alignment 结果。 |
| `data/demo/monomer/alignments/6KWC_1/hhsearch_output.hhr` | HHsearch template 搜索结果。 |
| `data/demo/monomer/inference.sh` | 使用上述 FASTA 和预计算 alignment 的单体推理示例脚本。 |

示例数据不包含 OpenFold 权重和 PDB mmCIF 模板库。运行 demo 前需要准备：

- `weight/openfold.pt`，或通过 `CHECKPOINT_PATH=/path/to/openfold.pt` 指定权重。
- PDB mmCIF 模板目录，默认读取 `data/databases/pdb_mmcif/mmcif_files`，也可通过 `TEMPLATE_MMCIF_DIR=/path/to/mmcif_files` 指定。

运行示例：

```bash
bash data/demo/monomer/inference.sh
```

如需改用其他设备，可设置 `MODEL_DEVICE`：

```bash
MODEL_DEVICE=cuda:0 bash data/demo/monomer/inference.sh
```

## 推理示例

```bash
python scripts/inference.py /path/to/fasta_dir /path/to/template_mmcif_dir \
  --output_dir ./outputs/inference \
  --config_preset model_1_ptm \
  --model_device cuda:0 \
  --use_precomputed_alignments /path/to/alignments \
  --openfold_checkpoint_path ./weight/openfold.pt
```

如果使用 AlphaFold JAX 参数：

```bash
python scripts/inference.py /path/to/fasta_dir /path/to/template_mmcif_dir \
  --output_dir ./outputs/inference \
  --config_preset model_1_ptm \
  --model_device cuda:0 \
  --use_precomputed_alignments /path/to/alignments \
  --jax_param_path ./weight/params_model_1_ptm.npz
```

## 训练示例

```bash
python scripts/train.py \
  /path/to/train_mmcif \
  /path/to/train_alignments \
  /path/to/template_mmcif \
  ./outputs/train \
  2021-10-10 \
  --config_preset initial_training \
  --max_epochs 1 \
  --train_epoch_len 1 \
  --gpus 1
```

多卡训练可使用 `torchrun` 启动，具体参数按运行环境调整。

## 数据和权重下载

OpenFold 权重：

```bash
bash scripts/download_openfold_params.sh ./weight
```

HuggingFace 权重：

```bash
bash scripts/download_openfold_params_huggingface.sh ./weight
```

PDB mmCIF 模板库：

```bash
bash scripts/download_pdb_mmcif.sh /path/to/database_dir
```

完整 AlphaFold/OpenFold 数据库：

```bash
bash scripts/download_alphafold_dbs.sh /path/to/database_dir full_dbs
```

## OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 引用与许可证

OpenFold 原始代码使用 Apache License 2.0。本仓库保留来源说明，并面向 OneScience ModelScope/OneCode 调用场景进行整理。

如果在科研工作中使用 OpenFold 结果，建议引用 OpenFold 原始论文、AlphaFold 相关论文和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。
