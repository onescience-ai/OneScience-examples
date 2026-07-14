<p align="center">
  <strong>
    <span style="font-size: 30px;">Protenix</span>
  </strong>
</p>

# 模型介绍

Protenix 是面向蛋白质、核酸和配体等生物分子复合物结构预测的 AlphaFold3-like 模型，可从 JSON 描述的分子输入和本地 MSA 特征预测三维结构，并输出 CIF 结构文件与置信度结果。当前 ModelScope 包面向下载即用、本地快速验证和 OneCode 自动化运行场景，代码、配置、示例输入和预训练权重均已放在当前目录内。

# 仓库说明

本仓库是 OneScience 整理的 Protenix 全量可运行模型仓库。`examples/` 和 `weight/` 已随模型包完整提供.

当前支持能力：

- 生物分子复合物结构推理，默认使用 `examples/7r6r.json` 和本地小型 MSA 示例。
- 单卡训练入口 `scripts/train.py`，用户需自行准备完整 Protenix 训练数据。
- 微调入口 `scripts/finetune.py`，默认加载 `weight/model_v0.5.0.pt` 和 `ft_datasets/finetune_subset.txt`。
- 包完整性预检 `scripts/preflight.py`，可检查权重、示例输入、配置路径和本地 `models.*` 导入。

当前不支持能力：

- 不内置完整训练数据集、CCD 缓存、MSA 数据库或任意新样本的 MSA 搜索数据库。
- 不在运行时自动访问远端补齐缺失文件；请保持模型包下载完整。
- 不提供独立标准评测入口、结构可视化服务或部署服务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 复合物结构预测 | 输入 Protenix JSON 和本地 MSA，输出预测 CIF 与置信度 JSON。 |
| ModelScope 全量包验证 | 使用包内 `config/ models/ scripts/ examples/ weight/` 布局直接预检和推理。 |
| 微调链路验证 | 使用包内权重和 `ft_datasets/finetune_subset.txt` 启动微调入口。 |
| 训练接口整理 | 用户提供完整 Protenix 数据集后，可运行单卡训练脚本。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | ModelScope 元信息 | 声明默认权重和默认输入 |
| `config/` | 推理和训练配置 | 默认读取 `examples/7r6r.json` 和 `weight/model_v0.5.0.pt` |
| `models/` | Protenix 所需源码依赖 | 已改为本地 `models.*` 导入 |
| `scripts/run_inference.py` | 推荐推理入口 | 默认输出到 `output_unified/` |
| `scripts/inference.py` | 推理核心脚本 | 由 `run_inference.py` 调用 |
| `scripts/train.py` | 训练入口 | 需要完整 Protenix 数据集 |
| `scripts/finetune.py` | 微调入口 | 默认加载本地预训练权重 |
| `scripts/preflight.py` | 包完整性预检 | `--strict-weights` 可校验真实权重 |
| `examples/` | 默认推理输入与小型 MSA | 已随 ModelScope 模型包提供 |
| `weight/` | Protenix v0.5.0 权重目录 | 已随 ModelScope 模型包提供 |
| `ft_datasets/` | 微调子集列表 | 默认包含 `finetune_subset.txt` |
| `MODEL_FILE_MANIFEST.tsv` | 模型文件清单 | 记录权重和示例输入文件信息 |

# 权重和数据准备

当前 ModelScope 包内已包含默认示例输入和 Protenix 预训练权重，下载完整模型包后可以直接使用。

当前 `weight/` 中应包含以下真实权重，文件名需保持不变：

```text
weight/model_v0.5.0.pt
```

默认示例输入包括：

```text
examples/7r6r.json
examples/7r6r/msa/1/pairing.a3m
examples/7r6r/msa/1/non_pairing.a3m
```

推理、训练和微调仍需要外部 Protenix 数据根目录提供 CCD 缓存和数据集文件。默认数据根目录为 `../bio_protenix_dataset`，可通过环境变量覆盖：

```bash
export DATA_ROOT_DIR=../bio_protenix_dataset
```

完整数据集至少需要：

```text
components.v20240608.cif
components.v20240608.cif.rdkit_mol.pkl
seq_to_pdb_index.json
indices/
mmcif_msa/
```

# 使用说明

## 1. OneCode 使用

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

## 快速开始

### 1. 安装onescience库

```bash
git clone https://gitee.com/onescience-ai/onescience
cd onescience
bash install.sh bio
```

### 2. 下载权重

```bash
bash download_assets.sh
```

### 3. 运行预检

仅检查模型包、权重和本地导入：

```bash
python scripts/preflight.py --strict-weights --strict-imports
```

如果已经准备完整数据集：

```bash
export DATA_ROOT_DIR=../bio_protenix_dataset
python scripts/preflight.py --strict-weights --strict-imports --strict-data
```

### 4. 运行推理

```bash
export DATA_ROOT_DIR=../bio_protenix_dataset
python scripts/run_inference.py
```

可用命令行参数覆盖默认采样设置：

```bash
python scripts/run_inference.py --sample_diffusion.N_sample 1 --sample_diffusion.N_step 20 --model.N_cycle 4
```

默认输出目录：

```text
output_unified/7r6r/seed_101/predictions/
```

### 5. 训练与微调

训练：

```bash
export DATA_ROOT_DIR=../bio_protenix_dataset
python scripts/train.py
```

微调：

```bash
export DATA_ROOT_DIR=../bio_protenix_dataset
python scripts/finetune.py
```

不下载外部训练数据库时，可以只做离线 smoke test。该模式会初始化训练/微调脚本、模型、优化器、损失函数，并在微调场景加载本地权重，但会跳过训练数据集初始化和训练循环：

```bash
PROTENIX_SMOKE_TEST=1 python scripts/train.py --max_steps 1 --use_wandb false
PROTENIX_SMOKE_TEST=1 python scripts/finetune.py --max_steps 1 --use_wandb false
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

Protenix 原始代码和权重请遵守其上游项目许可证、模型权重使用条款以及相关数据源要求。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

如果在科研工作中使用 Protenix 结果，建议引用 Protenix 原始论文和 OneScience 相关项目信息，并根据实际任务补充 wwPDB、CCD、MSA 数据库及下游分析工具引用。
