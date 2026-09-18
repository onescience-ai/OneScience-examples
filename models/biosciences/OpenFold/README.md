<p align="center">
  <strong>
    <span style="font-size: 30px;">OpenFold</span>
  </strong>
</p>

# 模型介绍

OpenFold 是一个基于蛋白质序列、多序列比对（MSA）和模板信息来预测蛋白质三维结构的开源框架。作为 AlphaFold2 的可训练复现，其核心网络结构包括 Evoformer、Structure Module、模板模块和 MSA 模块等，支持训练、推理及权重转换等完整的结构预测实验流程。

论文: OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization
https://www.biorxiv.org/content/10.1101/2022.11.20.517210v2

# 模型描述

OpenFold 基于 Evoformer + Structure Module 架构，使用 PDB 与 UniRef 数据进行训练，面向蛋白质三维结构预测、模型微调及架构改进研究。核心网络包含 48 层 Evoformer Block 与 8 层 Structure Module，通过三角注意力与不变点注意力机制实现进化信息与空间几何的双向融合；相比原版 AlphaFold2，新增内存高效注意力、混合精度训练支持及权重无缝转换能力，显著降低显存占用并修复可训练性缺陷。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 本地 OpenFold 调用 | 用户通过本仓库脚本调用 OpenFold 训练或推理。 |
| OneScience 基座复用 | 继续使用 OneScience 的 datapipes、utils、loss、np 等公共模块。 |
| ModelScope/OneCode 发布 | 只暴露脚本、模型、配置和权重目录，减少发布包体积。 |
| 权重转换和数据准备 | 使用 `scripts/` 下工具下载数据库、预处理 alignment、转换权重。 |


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

### 下载模型包

```bash
modelscope download --model OneScience/OpenFold --local_dir ./OpenFold
cd OpenFold
```

### 安装运行环境

#### DCU环境
```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
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
bash scripts/download_openfold_params.sh ./weights
```

HuggingFace 权重：

```bash
bash scripts/download_openfold_params_huggingface.sh ./weights
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

- OpenFold 原始论文：[OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization](https://www.biorxiv.org/content/10.1101/2022.11.20.517210)。

- AlphaFold2 原始论文：[Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)。

- 如果使用 OpenFold 的多聚体预测功能，还应引用 AlphaFold-Multimer 原始论文：[Protein complex prediction with AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)。

- 如果使用 OpenProteinSet 训练数据，还应引用：[OpenProteinSet: Training data for structural biology at scale](https://arxiv.org/abs/2308.05326)。

- OpenFold 相关源码使用 Apache License 2.0，详见仓库根目录 `LICENSE`。模型权重和数据的使用条款请以对应发布方说明为准。

- OpenProteinSet 数据集使用 CC BY 4.0 License。使用该数据集时，应注明数据来源，并按照数据集页面要求引用 OpenProteinSet 论文。

- 如果在科研工作中使用 OpenFold，建议同时引用 OpenFold 和 AlphaFold2 原始论文以及OneScience相关项目信息；使用多聚体功能时补充引用 AlphaFold-Multimer；使用 OpenProteinSet 或其他下游数据资源时，还应补充相应数据集、数据库和原始论文的引用。