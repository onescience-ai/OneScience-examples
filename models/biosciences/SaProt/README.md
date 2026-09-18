<p align="center">
  <strong>
    <span style="font-size: 30px;">SaProt</span>
  </strong>
</p>

# 模型介绍

SaProt（Protein Language Modeling with Structure-aware Vocabulary）是一类将蛋白质氨基酸序列与结构信息联合建模的蛋白质语言模型。其核心思想是将氨基酸（AA）与 Foldseek 生成的 3Di 结构字母组合成结构感知 token，使模型能够同时利用蛋白质序列与结构上下文进行表征学习。

SaProt 可用于蛋白质表征提取、零样本突变效应预测、蛋白质逆折叠以及下游任务微调等场景。

论文：

> **SaProt: Protein Language Modeling with Structure-aware Vocabulary**  
> ICLR 2024 Spotlight  
> 后续工作发表于 Nature Biotechnology（2025）

# 模型描述

SaProt 使用由氨基酸 AA 与 Foldseek 3Di 结构字母组合得到的结构感知词表进行蛋白质建模。

例如一个结构感知序列可表示为：

```text
M#EvVpQpL#VyQdYaKv
```

其中每两个字符组成一个结构感知 token，第一个字符表示氨基酸，第二个字符表示对应的 3Di 结构状态；`#` 可用于屏蔽低置信度结构区域。

官方提供多个规模的预训练模型：

| 模型 | 参数规模 | 训练数据 |
| --- | ---: | --- |
| `SaProt_35M_AF2` | 35M | 40M AF2 structures |
| `SaProt_650M_PDB` | 650M | 40M AF2 structures + 60K PDB structures |
| `SaProt_650M_AF2` | 650M | 40M AF2 structures |
| `SaProt_1.3B_AF2` | 1.3B | 40M AF2 structures |
| `SaProt_1.3B_AFDB_OMG_NCBI` | 1.3B | AFDB + OMG_prot50 + NCBI |

对于 35M 和 650M SaProt，官方建议使用带结构信息的 SA token 输入以获得最佳效果；1.3B 版本同时能够较好处理结构感知序列和仅氨基酸序列。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 蛋白质表征提取 | 提取 residue-level 或 protein-level embedding |
| 零样本突变效应预测 | 不进行任务微调，直接评估单点或多点突变 |
| 结构感知蛋白质建模 | 联合使用氨基酸与 3Di 结构 token |
| 蛋白质逆折叠 | 根据结构信息进行序列设计 |
| 下游任务微调 | 用于 EC、GO、稳定性、PPI、Contact、DeepLoc 等任务 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- SaProt 支持 CPU 和 GPU/DCU 推理。
- 35M 模型可用于轻量级测试；650M 和 1.3B 模型建议使用 GPU/DCU。
- 进行批量 embedding、突变扫描、预训练或微调时，显存和主机内存需求会明显增加。

### 安装运行环境

#### DCU环境

```bash
# 请首先激活 DTK 及 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

pip install onescience[bio] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

#### 环境说明

- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考 `requirements.txt` 声明的依赖版本，补充安装或调整相应依赖。

### 模型与数据准备

#### 1）SaProt 模型权重

官方模型主要发布在 Hugging Face：

```text
SaProt_35M_AF2
https://huggingface.co/westlake-repl/SaProt_35M_AF2

SaProt_650M_PDB
https://huggingface.co/westlake-repl/SaProt_650M_PDB

SaProt_650M_AF2
https://huggingface.co/westlake-repl/SaProt_650M_AF2

SaProt_1.3B_AF2
https://huggingface.co/westlake-repl/SaProt_1.3B_AF2

SaProt_1.3B_AFDB_OMG_NCBI
https://huggingface.co/westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI
```

以 `SaProt_650M_AF2` 为例，可提前离线下载：

```bash
huggingface-cli download \
  westlake-repl/SaProt_650M_AF2 \
  --local-dir ./weight/PLMs/SaProt_650M_AF2
```

建议统一将模型权重下载至 `weight/PLMs/` 下。当前 SaProt 配置默认读取：

```text
weight/PLMs/SaProt_650M_AF2
```

如果运行 ESM2 对照实验，还需要准备：

```bash
huggingface-cli download \
  facebook/esm2_t33_650M_UR50D \
  --local-dir ./weight/PLMs/esm2_t33_650M_UR50D
```

对应配置默认读取：

```text
weight/PLMs/esm2_t33_650M_UR50D
```

#### 2）Foldseek

SaProt 的结构感知输入需要先将 PDB/CIF 结构编码为 Foldseek 3Di 序列，官方 README 提供的下载地址为：

```text
https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view
```

也可以使用 Foldseek 官方 Linux 预编译包或系统/平台已安装的 Foldseek。

当前适配版建议将 Foldseek 放到：

```text
SaProt/
└── scripts/
    └── bin/
        └── foldseek
```

并赋予执行权限：

```bash
chmod +x scripts/bin/foldseek
```

当前配置文件中使用的 Foldseek 路径为：

```text
scripts/bin/foldseek
```

#### 3）下游任务数据集

官方下游任务数据集下载地址为：

```text
https://drive.google.com/drive/folders/11dNGqPYfLE3M-Mbh4U7IQpuHxJpuRr4g?usp=sharing
```

当前适配版建议将下游任务数据解压到：

```text
scripts/LMDB/
```

配置文件默认使用的典型路径包括：

```text
scripts/LMDB/Thermostability/foldseek/train
scripts/LMDB/Thermostability/foldseek/valid
scripts/LMDB/Thermostability/foldseek/test
scripts/LMDB/ProteinGym/substitutions
scripts/LMDB/ClinVar
```

#### 4）预训练数据集

重新预训练 SaProt 时，需要准备官方预训练数据：

```text
westlake-repl/AF2_UniRef50
https://huggingface.co/datasets/westlake-repl/AF2_UniRef50
```

官方预训练配置默认使用类似：

```text
scripts/LMDB/AF2_Uniref50/foldseek/train
scripts/LMDB/AF2_Uniref50/foldseek/valid
```

的 LMDB 数据目录。预训练数据集体量较大，只有在需要从头预训练或继续预训练时才需要准备。

## 3. 快速开始

### 下载模型包

```bash
modelscope download \
  --model OneScience/SaProt \
  --local_dir ./SaProt

cd SaProt
```

### 快速验证

检查依赖：

```bash
python - <<'PY'
import torch
import transformers
import esm
import pytorch_lightning as pl

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("pytorch_lightning:", pl.__version__)
print("SaProt dependencies OK")
PY
```

检查 Foldseek：

```bash
./scripts/bin/foldseek version
```

测试模型加载：

```bash
python - <<'PY'
from transformers import EsmTokenizer, EsmForMaskedLM

model_path = "./weight/PLMs/SaProt_650M_AF2"

tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)

print("SaProt load OK")
PY
```

# 示例数据

官方仓库提供：

```text
scripts/example/8ac8.cif
```

可用于演示蛋白质结构向结构感知序列的转换。

用户自己的结构输入可以为：

```text
*.pdb
*.cif
```

如果已经拥有经过 Foldseek 编码的结构感知序列，也可以直接送入 SaProt，而不需要再次处理结构文件。

# 推理示例

## 加载 SaProt 进行前向推理

```bash
python - <<'PY'
import torch
from transformers import EsmTokenizer, EsmForMaskedLM

model_path = "weight/PLMs/SaProt_650M_AF2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)
model.to(device)
model.eval()

seq = "M#EvVpQpL#VyQdYaKv"
tokens = tokenizer.tokenize(seq)
print(tokens)

inputs = tokenizer(seq, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.logits.shape)
PY
```

## 使用 ESM 接口加载 SaProt

如果模型目录中包含 `SaProt_650M_AF2.pt`，可以使用项目提供的 ESM 加载函数：

```bash
python - <<'PY'
from scripts.utils.esm_loader import load_esm_saprot

model_path = "weight/PLMs/SaProt_650M_AF2/SaProt_650M_AF2.pt"
model, alphabet = load_esm_saprot(model_path)

print("ESM SaProt load OK")
PY
```

## 结构文件转换为结构感知序列

```bash
python - <<'PY'
from scripts.utils.foldseek_util import get_struc_seq

pdb_path = "scripts/example/8ac8.cif"

parsed_seqs = get_struc_seq("scripts/bin/foldseek", pdb_path, ["A"], plddt_mask=False)["A"]
seq, foldseek_seq, combined_seq = parsed_seqs

print(f"seq: {seq}")
print(f"foldseek_seq: {foldseek_seq}")
print(f"combined_seq: {combined_seq}")
PY
```

其中 `["A"]` 表示只提取结构文件中的 A 链；返回结果中的 `combined_seq` 是 SaProt 可直接使用的结构感知序列。

## 突变效应预测

```bash
python - <<'PY'
import torch
from model.saprot.saprot_foldseek_mutation_model import SaprotFoldseekMutationModel

config = {
    "foldseek_path": None,
    "config_path": "weight/PLMs/SaProt_650M_AF2",
    "load_pretrained": True,
}
model = SaprotFoldseekMutationModel(**config)
tokenizer = model.tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

seq = "M#EvVpQpL#VyQdYaKv"

mut_info = "V3A"
mut_value = model.predict_mut(seq, mut_info)
print(mut_value)

mut_info = "V3A:Q4M"
mut_value = model.predict_mut(seq, mut_info)
print(mut_value)

mut_pos = 3
mut_dict = model.predict_pos_mut(seq, mut_pos)
print(mut_dict)

mut_pos = 3
mut_dict = model.predict_pos_prob(seq, mut_pos)
print(mut_dict)
PY
```

## 提取蛋白质 embedding

```bash
python - <<'PY'
import torch
from model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer

config = {
    "task": "base",
    "config_path": "weight/PLMs/SaProt_650M_AF2",
    "load_pretrained": True,
}

model = SaprotBaseModel(**config)
tokenizer = EsmTokenizer.from_pretrained(config["config_path"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

seq = "M#EvVpQpL#VyQdYaKv"
tokens = tokenizer.tokenize(seq)
print(tokens)

inputs = tokenizer(seq, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    embeddings = model.get_hidden_states(inputs, reduction="mean")

print(embeddings[0].shape)
PY
```

## 蛋白质逆折叠

逆折叠需要额外准备逆折叠模型权重：

```text
https://huggingface.co/westlake-repl/SaProt_650M_AF2_inverse_folding
```

下载后建议放到：

```text
weight/PLMs/SaProt_650M_AF2_inverse_folding
```

运行示例：

```bash
python - <<'PY'
import torch
from model.saprot.saprot_if_model import SaProtIFModel

config = {
    "config_path": "weight/PLMs/SaProt_650M_AF2_inverse_folding",
    "load_pretrained": True,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SaProtIFModel(**config)
model = model.to(device)

aa_seq = "##########"
struc_seq = "dddddddddd"

pred_aa_seq = model.predict(aa_seq, struc_seq)
print(pred_aa_seq)
PY
```

# 训练说明

本仓库使用统一训练入口：

```bash
python scripts/training.py -c <config_path>
```

配置文件位于 `conf/` 目录，模型代码位于 `model/`，数据处理与工具代码位于 `scripts/`。当前配置默认使用 `weight/PLMs/SaProt_650M_AF2` 中的预训练模型权重。

## 预训练

如需重新预训练或继续预训练 SaProt，需要先准备预训练 LMDB 数据集，然后运行：

```bash
python scripts/training.py -c conf/pretrain/saprot.yaml
```

该配置默认读取：

```text
scripts/LMDB/AF2_Uniref50/foldseek/train
scripts/LMDB/AF2_Uniref50/foldseek/valid
```

## 下游微调

以下命令用于在下游任务上微调 SaProt：

```bash
# Thermostability
python scripts/training.py -c conf/Thermostability/saprot.yaml

# EC
python scripts/training.py -c conf/EC/saprot.yaml

# GO
python scripts/training.py -c conf/GO/MF/saprot.yaml
python scripts/training.py -c conf/GO/BP/saprot.yaml
python scripts/training.py -c conf/GO/CC/saprot.yaml

# Metal ion binding
python scripts/training.py -c conf/MetalIonBinding/saprot.yaml

# Human PPI
python scripts/training.py -c conf/HumanPPI/saprot.yaml

# Contact prediction
python scripts/training.py -c conf/Contact/saprot.yaml

# DeepLoc
python scripts/training.py -c conf/DeepLoc/cls2/saprot.yaml
python scripts/training.py -c conf/DeepLoc/cls10/saprot.yaml
```

在单卡或小显存环境中，建议使用 `conf/scnet/Thermostability_saprot_1gpu.yaml` 作为起点：

```bash
python scripts/training.py -c conf/scnet/Thermostability_saprot_1gpu.yaml
```

## Zero-shot 突变效应评估

ProteinGym 评估：

```bash
python scripts/mutation_zeroshot.py -c conf/ProteinGym/saprot.yaml
```

输出文件默认保存到：

```text
output/ProteinGym/SaProt_650M_AF2.tsv
```

ClinVar 评估：

```bash
python scripts/mutation_zeroshot.py -c conf/ClinVar/saprot.yaml
python scripts/compute_clinvar_auc.py -c conf/ClinVar/saprot.yaml
```

ClinVar 预测结果默认保存到：

```text
output/ClinVar/SaProt_650M_AF2
```

单卡环境也可以使用适配配置：

```bash
python scripts/mutation_zeroshot.py -c conf/scnet/ClinVar_saprot.yaml
python scripts/compute_clinvar_auc.py -c conf/scnet/ClinVar_saprot.yaml
```

## ESM2 对照实验

如果需要运行 ESM2 baseline，需要额外准备 `weight/PLMs/esm2_t33_650M_UR50D` 权重以及对应的 normal 版 LMDB 数据。示例命令：

```bash
python scripts/training.py -c conf/Thermostability/esm2.yaml
python scripts/mutation_zeroshot.py -c conf/ProteinGym/esm2.yaml
```

# 输出说明

| 任务 | 主要输出 |
| --- | --- |
| 结构编码 | AA 序列、3Di 序列、结构感知序列 |
| 模型前向 | token-level logits |
| 蛋白质表征 | residue-level / protein-level embedding |
| 突变效应预测 | mutation score |
| Zero-shot 评估 | ProteinGym 的 Spearman 结果或 ClinVar 的 AUC 结果 |
| 逆折叠 | 给定结构条件下生成或评估的蛋白质序列 |
| 下游微调 | 对应任务预测结果及模型 checkpoint |

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- SaProt 官方源码仓库采用 **MIT License**，允许使用、修改、分发、再许可和商业使用；复制或分发时应保留原始版权声明和 MIT License 许可证文本。

- SaProt 模型权重通过 Hugging Face 独立发布，如需商业使用、再分发或其他用途，应分别核对并遵守对应模型页面的许可证；相关预训练数据集和下游数据集也应遵守各自数据集页面的许可与使用条款。

- 本仓库为 SaProt 的 **DCU 适配版本**，仓库代码、模型权重及相关数据的使用仍应以各自原始项目中的许可证及使用条款为准。

