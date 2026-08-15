<p align="center">
  <strong>
    <span style="font-size: 30px;">scGPT</span>
  </strong>
</p>

# 模型介绍

scGPT（single-cell Generative Pre-trained Transformer）是由多伦多大学 Bo Wang 团队研发的单细胞组学基础模型。模型在超过 3300 万个细胞的单细胞 RNA 测序数据上预训练，通过基因级 Token 化与生成式预训练学习细胞与基因的通用表征，可迁移到细胞类型注释、批次整合、多组学整合、扰动预测等下游任务。本项目包含细胞嵌入推理和细胞类型注释微调相关能力，支持单卡与多卡运行。

主要论文：

- scGPT 基础模型：scGPT: toward building a foundation model for single-cell multi-omics using generative AI  
  https://www.nature.com/articles/s41592-024-02201-0

# 仓库说明

本仓库是 scGPT 最小可运行独立模型仓库，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用 `scGPT_human` 预训练权重对兼容的 AnnData 文件执行细胞嵌入推理，输出 `obsm["X_scGPT"]`
- 在带标签的单细胞数据上微调细胞类型分类头，输出 `best_model.pt`、`args.json`、`vocab.json`、`metrics.json`
- 单卡与多卡自动分发（推理按细胞区间分片合并，微调使用 DDP）
- 使用少量细胞（`--max-cells`）快速验证数据读取、模型加载、训练与推理全链路

当前不支持能力：

- 不负责从头预训练（未提供预训练脚本与大规模预训练数据）
- 不包含扰动预测、多组学整合等实验性流程（OneScience state-transition 框架功能）
- 不面向临床诊断或医学决策

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 细胞嵌入推理 | 输入单细胞 AnnData 文件，输出 `obsm["X_scGPT"]` 细胞嵌入 |
| 细胞类型注释微调 | 输入带标签的单细胞数据，微调并输出模型权重与验证指标 |
| 替换预训练权重 | 通过 `SCGPT_MODEL_DIR` 切换 9 个已上传的预训练模型 |
| OneCode / 本地运行 | 在生物领域运行环境中快速验证脚本连通性 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `download_assets.sh` | 权重与数据集下载脚本 | 权重来自 `OneScience/scGPT`，数据集来自 `OneScience/scGPT_datasets` |
| `config/config.yaml` | 配置文件，声明模型、数据、输出目录与默认超参 | 相对路径相对本目录 |
| `model/` | scGPT 模型源码 | 自包含实现，不导入 onescience |
| `model/LICENSE.scgpt` | scGPT 开源许可证 | MIT License |
| `scripts/infer.sh` | 细胞嵌入推理入口 | 需先下载权重并准备推理数据 |
| `scripts/finetune.sh` | 细胞类型注释微调入口 | 需先下载权重并准备训练数据 |
| `scripts/embed.py` | 细胞嵌入推理脚本 | 数据管线等通过已安装的 onescience 导入 |
| `scripts/finetune.py` | 细胞类型注释微调脚本 | 数据管线等通过已安装的 onescience 导入 |
| `scripts/_scgpt_common.sh` | Shell 公共路径与设备检测 | 所有入口共享 |
| `weight/` | 权重目录 | 由 `download_assets.sh` 下载，不提交 Git |
| `data/` | 数据集目录 | 由 `download_assets.sh` 下载，不提交 Git |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**软件要求**

请参考 OneScience 生物领域运行环境，DCU 用户想了解更多适配内容请联系 liubiao@sugon.com。

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光 DCU：

```bash
hy-smi
```

## 3. 快速开始

本模型包不包含权重文件，需要先从 ModelScope 下载预训练权重。

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

```bash
# 如果需要找不到库的情况需要激活cuda，参考下列代码
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
```

安装完成后回到模型包目录：

```bash
cd ./scGPT
```

### 下载数据集&权重

```bash
bash download_assets.sh
```

脚本完成两项下载并逐文件校验完整性：

- 权重：从 ModelScope 模型仓库 `OneScience/scGPT` 下载 `weight/**`（9 个预训练模型，每个包含 `args.json`、`best_model.pt`、`vocab.json`）到 `weight/`。
- 数据集：从 ModelScope 数据集仓库 `OneScience/scGPT_datasets`（CC BY 4.0）下载演示数据到 `data/`，逐文件来源与引用见该仓库中的 `DATA_SOURCES.md`。如只需权重，可设置 `SCGPT_SKIP_DATASET=1` 跳过。

也可以手动下载：

```bash
modelscope download --model OneScience/scGPT --include "weight/**" --local_dir ./scGPT
modelscope download --dataset OneScience/scGPT_datasets --local_dir ./scGPT/data
```

默认模型目录为 `weight/scGPT_human`，可通过环境变量切换其它权重：

```bash
export SCGPT_MODEL_DIR=weight/scGPT_pan_cancer
```

### 准备数据

数据通过环境变量指定，无需修改脚本：

| 环境变量 | 默认值 | 说明 |
| :---: | :---: | :---: |
| `SCGPT_MODEL_DIR` | `weight/scGPT_human` | 包含 `args.json`、`best_model.pt`、`vocab.json` 的模型目录 |
| `SCGPT_DATASET_ROOT` | `data` | 数据集根目录 |
| `SCGPT_INFERENCE_DATA` | `data/annotation_pancreas/demo_test.h5ad` | 嵌入推理输入 |
| `SCGPT_FINETUNE_DATA` | `data/annotation_pancreas/demo_train.h5ad` | 微调输入 |
| `SCGPT_OUTPUT_ROOT` | `outputs` | 输出根目录 |
| `SCGPT_DEVICE` | `cuda` | PyTorch 计算设备 |

演示数据由 `download_assets.sh` 下载到 `data/`，默认输入为：

```text
data/annotation_pancreas/demo_test.h5ad
data/annotation_pancreas/demo_train.h5ad
```

如使用共享运行目录中的数据，也可以通过环境变量指定：

```bash
export SCGPT_INFERENCE_DATA=${ONESCIENCE_DATASETS_DIR}/scGPT/annotation_pancreas/demo_test.h5ad
export SCGPT_FINETUNE_DATA=${ONESCIENCE_DATASETS_DIR}/scGPT/annotation_pancreas/demo_train.h5ad
```

### 细胞嵌入推理

```bash
bash scripts/infer.sh
```

默认输出文件为 `outputs/pancreas_embeddings.h5ad`，归一化后的细胞嵌入保存在 `obsm["X_scGPT"]` 中。可以在 Bash 命令后追加 Python 脚本支持的参数。例如，仅对 64 个细胞执行推理：

```bash
bash scripts/infer.sh \
  --max-cells 64 \
  --max-length 256 \
  --batch-size 8 \
  --output outputs/pancreas_embeddings_64.h5ad
```

### 细胞类型注释微调

共享胰腺数据集使用 `Celltype` 作为标签列，使用 `Gene Symbol` 作为基因符号列，脚本已将其设为默认值：

```bash
bash scripts/finetune.sh \
  --epochs 5 \
  --batch-size 32
```

默认输出目录为 `outputs/pancreas_finetune`，其中包含：

- `best_model.pt`：验证指标最优的模型权重。
- `args.json`：模型配置、标签名称和数据处理信息。
- `vocab.json`：模型使用的基因词表。
- `metrics.json`：验证集指标。

对于元数据字段不同的数据集，可以使用 `--gene-column` 或 `--label-column` 指定相应列。未指定基因列时，程序会依次检查 `Gene Symbol`、`feature_name`、`gene_name`、`gene_symbols` 和 `symbol`，均不存在时使用 AnnData 的 `var_names`。可以使用 `--data-is-raw` 或 `--data-is-normalized` 显式指定表达矩阵为原始计数或已归一化数据。

### 单卡与多卡运行

推理和微调入口会通过 `torch.cuda.device_count()` 自动检测可见计算设备数量：单卡直接运行，多卡自动使用 `torchrun` 每卡一个进程。

- 多卡推理按连续的细胞区间将数据分配给各张卡，最后由主进程按原始顺序合并嵌入。
- 多卡微调使用 DistributedDataParallel 和 DistributedSampler，`--batch-size` 表示每张卡的批量大小（例如 8 张卡且 `--batch-size 4` 时，有效全局批量大小为 32），`--max-steps` 表示所有 rank 同步执行的优化步数。

### 注意力后端

默认使用 PyTorch 注意力实现，以保证检查点兼容性。只有在确认已安装的 Flash Attention 扩展与当前运行时匹配时，才建议设置以下环境变量并为嵌入推理传入 `--use-fast-transformer`：

```bash
export ONESCIENCE_SCGPT_ENABLE_FLASH_ATTN=1
bash scripts/infer.sh --use-fast-transformer
```

微调示例默认使用 PyTorch 注意力实现。

# 数据格式

scGPT 使用 AnnData：

```text
*.h5ad
```

嵌入推理输入至少需要：

```text
adata.X              基因表达矩阵
adata.var[基因名列]   基因符号（默认依次检查 Gene Symbol、feature_name、gene_name、gene_symbols、symbol，均不存在时使用 var_names）
```

微调输入至少需要：

```text
adata.X                   基因表达矩阵（原始计数或已归一化数据均可，可用 --data-is-raw / --data-is-normalized 指定）
adata.obs[标签列]          细胞类型标签（默认 Celltype，可用 --label-column 指定）
```

演示数据集由 ModelScope 数据集仓库 `OneScience/scGPT_datasets` 提供，以 CC BY 4.0 许可发布，逐文件来源与引用见该仓库中的 `DATA_SOURCES.md`。

# 验证

进行较长时间的训练前，可以先运行以下短流程验证完整链路：

```bash
bash scripts/infer.sh \
  --max-cells 64 \
  --max-length 256 \
  --batch-size 8 \
  --output outputs/pancreas_embeddings_64.h5ad

bash scripts/finetune.sh \
  --max-cells 64 \
  --n-hvg 200 \
  --max-length 201 \
  --batch-size 8 \
  --epochs 1 \
  --max-steps 2 \
  --freeze-encoder \
  --output-dir outputs/pancreas_finetune_64
```

语法检查：

```bash
python -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for root in ['model', 'scripts'] for p in pathlib.Path(root).rglob('*.py')]"
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- scGPT 原始论文：scGPT: toward building a foundation model for single-cell multi-omics using generative AI。
- 论文地址：https://www.nature.com/articles/s41592-024-02201-0
- 原始项目：https://github.com/bowang-lab/scGPT
- scGPT 相关源码使用 MIT License，见 `model/LICENSE.scgpt`。模型权重和数据的使用条款请以对应发布方说明为准。
- 如果在科研工作中使用 scGPT 结果，建议引用：

```bibtex
@article{cui2024scgpt,
  title = {scGPT: toward building a foundation model for single-cell multi-omics using generative AI},
  author = {Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Duan, Nan and Wang, Bo},
  journal = {Nature Methods},
  year = {2024},
  doi = {10.1038/s41592-024-02201-0},
}
```
