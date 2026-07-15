
<p align="center">
  <strong>
    <span style="font-size: 30px;">La-Proteina</span>
  </strong>
</p>

# 模型介绍

La-Proteina 是一种基于**部分隐变量流匹配（Partially Latent Flow Matching）**的蛋白质结构生成模型，能够直接生成全原子蛋白质结构及其对应的氨基酸序列。模型将蛋白质的骨架（backbone CA）显式建模，而序列和原子级细节则通过每个残基的固定维度隐变量来捕捉，从而有效避免显式侧链表示带来的挑战。

论文：_La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching_（arXiv 2025）。

- [论文链接](https://arxiv.org/abs/2507.09466)
- [项目主页](https://research.nvidia.com/labs/genair/la-proteina/)
- [Model Card++](./modelcard/model_card_overview.md)

La-Proteina 在多个生成基准上取得了 state-of-the-art 性能，包括全原子协同可设计性（co-designability）、多样性、结构有效性以及原子级 motif 支架（motif scaffolding）。模型可生成最长约 800 个残基的蛋白质结构。

# 仓库说明

本示例将 La-Proteina 集成到 OneScience 生物信息（AI for Biology）组件中，提供训练、蛋白质结构生成、生成结果评估与自编码器推理的统一入口。

当前支持能力：

- **蛋白质结构生成**：基于流匹配生成蛋白质主链（backbone CA）与局部隐变量（local latents）。
- **Motif 约束生成**：支持 motif 位置与序列约束，实现功能 motif 的骨架设计。
- **训练扩散模型**：在 PDB 数据集上训练 La-Proteina 主模型。
- **训练自编码器**：训练局部隐变量自编码器（local-latent autoencoder）。
- **评估生成结果**：计算生成结构的 RMSD、序列恢复率、(co-)designability 等指标。
- **自编码器推理**：对 PDB 结构进行编码-解码重建并评估重建质量。

当前不支持能力：

- 当前仓库快照未打包 `dataset=genie2` 与 `dataset=pdb_multimer`，运行会显式报错。
- 当前仓库不内置proteinMPNN，使用La-Proteina进行评估时需要自行下载；

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 蛋白质结构生成 | 基于流匹配生成蛋白质主链（backbone CA）与局部隐变量（local latents）。 |
| Motif 约束生成 | 支持 motif 位置与序列约束，实现功能 motif 的骨架设计。 |
| 扩散模型训练 | 在 PDB 数据集上训练 La-Proteina 主模型。 |
| 自编码器训练与推理 | 训练局部隐变量自编码器，并对 PDB 结构执行编码、解码与重建评估。 |
| 生成结果评估 | 计算 RMSD、序列恢复率和 (co-)designability 等指标。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `scripts/run_train.sh` | 训练 La-Proteina 主模型 | 输出至 `./store/<run_name>/` |
| `scripts/run_generate.sh` | 蛋白质结构生成 | 支持无条件与 motif 约束生成 |
| `scripts/run_evaluate.sh` | 生成结构评估 | (co-)designability 需要 ProteinMPNN 权重 |
| `scripts/run_ae_infer.sh` | 自编码器编码、解码与重建 | 输出至 `./inference_ae/` |
| `train_laproteina.py` | 主模型训练入口 | 使用 Hydra 配置 |
| `infer_laproteina.py` | 蛋白质结构生成入口 | - |
| `evaluate_laproteina.py` | 评估入口 | - |
| `train_laproteina_ae.py` | 自编码器训练入口 | - |
| `infer_laproteina_ae.py` | 自编码器推理入口 | - |

```
/laproteina/
├── configs/                          # 配置文件
├── models/                          # 源码模块
├── scripts/                          # 可执行脚本（已提供）
    ├── run_train.sh                  # 训练 La-Proteina 主模型
    ├── run_generate.sh               # 蛋白质结构生成
    ├── run_evaluate.sh               # 生成结果评估
    └── run_ae_infer.sh               # 自编码器推理/重建
    ├── train_laproteina.py               # 训练入口（Hydra 配置）
    ├── infer_laproteina.py               # 生成入口
    ├── evaluate_laproteina.py            # 评估入口
    ├── train_laproteina_ae.py            # 自编码器训练入口
    ├── infer_laproteina_ae.py            # 自编码器推理入口
└── README.md                         # 本文档
```

# 使用说明

## 1. OneCode使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用GPU或DCU运行。
- CPU可以用于连通性验证，但速度较慢。
- DCU用户需要预先安装DTK，建议使用DTK 25.04.2以上版本或与当前集群匹配的OneScience推荐版本。

**软件要求**

DCU用户想了解更多适配内容请联系 liubiao@sugon.com

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光DCU：

```bash
hy-smi
```

### 环境准备

可选：通过环境变量覆盖默认路径：

    ```bash
    export LAPROTEINA_ROOT=/path/to/la-proteina
    export LAPROTEINA_DATASET_DIR=/path/to/dataset
    export LAPROTEINA_CHECKPOINTS_DIR=/path/to/checkpoints_laproteina
    export DATA_PATH=/path/to/dataset
    ```

## 快速开始

### 1. 安装运行环境

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```
#如果下述代码运行存在找不到库的情况，需要激活cuda，参考下列代码

```bash
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
```

### 2. 下载数据库(含权重)

```bash
modelscope download --dataset OneScience/La-Proteina --local_dir ./dataset
cd model
```

### 3. 使用方式

建议在 `laproteina` 目录下运行脚本，以便输出集中管理。

#### 1. 训练 La-Proteina 主模型（`run_train.sh`）

```bash
cd examples/biosciences/laproteina
bash scripts/run_train.sh
```

**常用 Hydra 参数覆盖：**

```bash
# 单卡调试
bash scripts/run_train.sh hardware.ngpus_per_node_=1 single=true

# 指定运行名称
bash scripts/run_train.sh run_name=my_laproteina_run

# 覆盖数据集或网络配置（对应 motif 训练场景）
bash scripts/run_train.sh dataset=pdb/pdb_train_motif_aa nn=local_latents_score_nn_160M_motif_idx_aa
```

输出：

- `./store/<run_name>/`：训练日志、检查点与 Hydra 配置


#### 2. 蛋白质结构生成（`run_generate.sh`）

```bash
cd examples/biosciences/laproteina
bash scripts/run_generate.sh
```

默认使用 `inference_ucond_tri` 配置进行无条件生成（LD2 模型 + AE1）。

**切换生成配置：**

```bash
# 无条件生成（无三角注意力）
bash scripts/run_generate.sh --config_name inference_ucond_notri

# 无条件生成长链（300-800 残基）
bash scripts/run_generate.sh --config_name inference_ucond_notri_long

# 索引式全原子 motif 支架
bash scripts/run_generate.sh --config_name inference_motif_idx_aa

# 索引式 tip-原子 motif 支架
bash scripts/run_generate.sh --config_name inference_motif_idx_tip

# 非索引式全原子 motif 支架
bash scripts/run_generate.sh --config_name inference_motif_uidx_aa

# 非索引式 tip-原子 motif 支架
bash scripts/run_generate.sh --config_name inference_motif_uidx_tip
```

输出：

- `./inference/<config_name>/`：生成的蛋白质结构文件与元数据

---

#### 3. 生成结果评估（`run_evaluate.sh`）

```bash
cd examples/biosciences/laproteina
bash scripts/run_evaluate.sh
```

默认评估 `inference_ucond_tri` 配置对应的生成结果。

**ProteinMPNN 权重准备：**

评估前需下载 ProteinMPNN 权重，可从魔搭社区下载：

```bash
modelscope download --model OneScience/ProteinMPNN --local_dir ./weight
```

输出：

- `./inference/<config_name>/evaluation/`：评估结果文件

---

#### 4. 自编码器推理（`run_ae_infer.sh`）

```bash
cd examples/biosciences/laproteina
bash scripts/run_ae_infer.sh
```

对 PDB 数据集执行编码-解码重建，评估重建指标（如全原子 RMSD、序列恢复率等）。

脚本会检查 `DATA_PATH/pdb_train` 与 `AE1_ucond_512.ckpt` 是否存在。

脚本内部调用：

```bash
python infer_laproteina_ae.py "$@"
```

常用参数覆盖：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LAPROTEINA_ROOT` | `${ONESCIENCE_DATASETS_DIR}/la-proteina` | 数据与权重根目录 |
| `LAPROTEINA_CHECKPOINTS_DIR` | `${LAPROTEINA_ROOT}/checkpoints_laproteina` | 自编码器权重目录 |
| `DATA_PATH` | `${LAPROTEINA_ROOT}/dataset` | 数据集目录 |

输出：

- `./inference_ae/`：重建结构与评估指标

### 注意事项

- 运行脚本前需确保 `ONESCIENCE_DATASETS_DIR` 环境变量已正确设置。
- 训练脚本默认检查 `DATA_PATH/pdb_train` 与 `AE1_ucond_512.ckpt` 是否存在，缺失会报错退出。
- 当前集成中 `dataset=pdb` 已可用，`dataset=genie2` 与 `dataset=pdb_multimer` 在当前仓库快照中未打包，运行会显式报错。
- 脚本会自动设置 ROCm/DCU 相关的 `LD_LIBRARY_PATH`，在海光 DCU 平台可直接运行；在 CUDA 平台可忽略或按需调整。
- 生成 motif 支架结构时，请确保 LD 模型与对应的 AE 模型配对正确，否则可能因长度/任务不匹配导致失败。
- 评估 (co-)designability 需要 ProteinMPNN 权重，请提前运行 `script_utils/download_pmpnn_weights.sh` 下载。
- 所有脚本建议在 `examples/biosciences/laproteina` 目录下执行，以便输出目录统一。
- 通过 Hydra 覆盖配置时，可以使用 `+CK_PATH=...` 指定 checkpoint 根路径，脚本在未提供时会自动设置为 `LAPROTEINA_ROOT`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

示例代码采用 Apache 2.0 许可证。La-Proteina 模型权重采用 [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)，其他材料采用 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)。

如果您在研究中使用了 La-Proteina，请引用原始论文：

```bibtex
@article{geffner2025laproteina,
  title={La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching},
  author={Geffner, Tomas and Didi, Kieran and Cao, Zhonglin and Reidenbach, Danny and Zhang, Zuobai and Dallago, Christian and Kucukbenli, Emine and Kreis, Karsten and Vahdat, Arash},
  journal={arXiv preprint arXiv:2507.09466},
  year={2025}
}
```
