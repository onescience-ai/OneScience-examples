<p align="center">
  <strong>
    <span style="font-size: 30px;">scGPT</span>
  </strong>
</p>

# 模型介绍

scGPT（single-cell Generative Pre-trained Transformer）是由多伦多大学 Bo Wang 团队研发的单细胞组学基础模型。模型在超过 3300 万个细胞的单细胞 RNA 测序数据上预训练，通过基因级 Token 化与生成式预训练学习细胞与基因的通用表征，可迁移到细胞类型注释、批次整合、多组学整合、扰动预测等下游任务。

论文：scGPT: toward building a foundation model for single-cell multi-omics using generative AI

https://www.nature.com/articles/s41592-024-02201-0

OneScience 完成了 scGPT 在 GPU/DCU 上的移植适配与合入，本仓库提供其中的细胞嵌入推理与细胞类型注释微调能力。


# 模型描述

scGPT 将基因表达矩阵转换为基因 Token 序列，以 Transformer 为主干，通过掩码值预测（MVC）、细胞嵌入对比（CCE）等生成式目标在大规模单细胞数据上预训练，输出细胞与基因两个层级的通用表征，并可挂接下游任务头（如细胞类型分类 CLS）进行微调。

本仓库的模型结构位于 `model/` 目录，为自包含实现；数据管线（datapipes）、评估指标（metrics）与训练/推理工具（utils）通过环境中已安装的 onescience 包提供。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 细胞嵌入推理 | 输入单细胞 AnnData（h5ad），输出 `obsm["X_scGPT"]` 细胞嵌入 |
| 细胞类型注释微调 | 在带标签的单细胞数据上微调，输出 `best_model.pt`、`args.json`、`vocab.json`、`metrics.json` |
| 本地快速验证 | 使用少量细胞（`--max-cells`）快速验证数据读取、模型加载、训练与推理全链路 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/scGPT --local_dir ./scGPT
cd scGPT
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install -r requirements.txt
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```


### 权重与数据准备

**权重**

将预训练模型目录（包含 `args.json`、`best_model.pt`、`vocab.json`）放到 `weight/scGPT_human`，或通过环境变量 `SCGPT_MODEL_DIR` 指定其它位置。

**数据**

将 `demo_test.h5ad`、`demo_train.h5ad` 放到 `data/annotation_pancreas/`，或通过以下环境变量覆盖具体位置，无需修改脚本：

| 环境变量 | 默认值 | 说明 |
| :---: | :---: | :---: |
| `SCGPT_MODEL_DIR` | `weight/scGPT_human` | 包含 `args.json`、`best_model.pt`、`vocab.json` 的模型目录 |
| `SCGPT_DATASET_ROOT` | `data` | 数据集根目录 |
| `SCGPT_INFERENCE_DATA` | `data/annotation_pancreas/demo_test.h5ad` | 嵌入推理输入 |
| `SCGPT_FINETUNE_DATA` | `data/annotation_pancreas/demo_train.h5ad` | 微调输入 |
| `SCGPT_OUTPUT_ROOT` | `outputs` | 输出根目录 |
| `SCGPT_DEVICE` | `cuda` | PyTorch 计算设备 |

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

进行较长时间的训练前，可以先运行以下短流程验证完整链路：

```bash
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


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- scGPT 原始代码使用 MIT License（详见 `model/LICENSE.scgpt`）。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

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

