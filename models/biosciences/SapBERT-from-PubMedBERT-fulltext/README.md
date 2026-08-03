---
license: apache-2.0
language:
- en
tags:
- biomedical
- lexical-semantics
- bionlp
- biology
- science
- embedding
- entity-linking
datasets:
- UMLS
---

**[News]** A cross-lingual extension of SapBERT appeared at the main conference of **ACL 2021**.  
**[News]** SapBERT appeared in the conference proceedings of **NAACL 2021**.

# SapBERT-from-PubMedBERT-fulltext

## Model description

SapBERT was proposed by [Liu et al. (2020)](https://arxiv.org/pdf/2010.11784.pdf).

The model was trained using the English portion of [UMLS 2020AA](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html), based on:

```text
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
```

SapBERT learns biomedical entity representations by aligning synonymous biomedical concepts in the embedding space.

It can be used for tasks such as:

- Biomedical entity representation
- Medical entity linking
- Biomedical semantic similarity
- Biomedical concept retrieval
- Synonym identification

## Expected input and output

The input is a list of biomedical entity names, for example:

```text
myocardial infarction
heart attack
COVID-19
Coronavirus infection
Hydroxychloroquine
```

The output is the `[CLS]` embedding from the last hidden layer of SapBERT.

The test script included in this repository extracts these entity embeddings and calculates pairwise cosine similarities.

## Original embedding example

The following example converts a list of biomedical entity names into embeddings:

```python
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)

model = AutoModel.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
).cuda()

all_names = [
    "covid-19",
    "Coronavirus infection",
    "high fever",
    "Tumor of posterior wall of oropharynx",
]

batch_size = 128
all_embeddings = []

for start_index in tqdm(
    np.arange(
        0,
        len(all_names),
        batch_size,
    )
):
    tokens = tokenizer.batch_encode_plus(
        all_names[
            start_index:
            start_index + batch_size
        ],
        padding="max_length",
        max_length=25,
        truncation=True,
        return_tensors="pt",
    )

    tokens_cuda = {
        key: value.cuda()
        for key, value in tokens.items()
    }

    with torch.inference_mode():
        cls_embeddings = model(
            **tokens_cuda
        )[0][:, 0, :]

    all_embeddings.append(
        cls_embeddings
        .cpu()
        .numpy()
    )

all_embeddings = np.concatenate(
    all_embeddings,
    axis=0,
)
```

For more details about training and evaluation, see the original [SapBERT GitHub repository](https://github.com/cambridgeltl/sapbert).

---

# 本地模型下载与测试

本代码仓库只保存模型配置、下载脚本和测试脚本，不直接保存体积较大的词表和模型权重文件。

## 1. 下载前的目录结构

克隆代码仓库后，模型目录结构如下：

```text
SapBERT-from-PubMedBERT-fulltext/
├── config/
│   ├── config.json
│   ├── special_tokens_map.json
│   └── tokenizer_config.json
├── scripts/
│   ├── download_weights.py
│   └── test.py
├── weight/
│   └── .gitkeep
└── README.md
```

代码仓库中不直接包含以下文件：

```text
config/vocab.txt
weight/model.safetensors
```

这两个文件需要通过下载脚本从 Hugging Face 获取。

## 2. 环境依赖

模型测试需要以下 Python 库：

```text
torch
transformers
safetensors
numpy
tqdm
huggingface_hub
```

可以先检查当前环境中是否已经安装：

```bash
python -c "import torch, transformers, safetensors, numpy, tqdm, huggingface_hub; print('所有依赖均已安装')"
```

检查 Hugging Face Hub 的版本：

```bash
python -c "import huggingface_hub; print('huggingface_hub版本：', huggingface_hub.__version__)"
```

如果没有安装 `huggingface_hub`，运行：

```bash
python -m pip install -U huggingface_hub
```

如果缺少其他依赖，可以运行：

```bash
python -m pip install -U transformers safetensors numpy tqdm
```

请根据当前 CUDA、DCU 或 CPU 环境安装合适版本的 PyTorch，不要在已有专用 PyTorch 环境中随意覆盖安装。

## 3. 下载词表和模型权重

进入 SapBERT 模型根目录后运行：

```bash
python scripts/download_weights.py
```

下载脚本会从以下 Hugging Face 仓库获取文件：

```text
cambridgeltl/SapBERT-from-PubMedBERT-fulltext
```

脚本会下载：

```text
vocab.txt
model.safetensors
```

并分别保存到：

```text
config/vocab.txt
weight/model.safetensors
```

下载完成后的目录结构如下：

```text
SapBERT-from-PubMedBERT-fulltext/
├── config/
│   ├── config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── scripts/
│   ├── download_weights.py
│   └── test.py
├── weight/
│   ├── .gitkeep
│   └── model.safetensors
└── README.md
```

当前测试流程只需要：

```text
model.safetensors
```

原始 Hugging Face 仓库中的以下其他框架权重不属于当前测试的必需文件：

```text
pytorch_model.bin
tf_model.h5
flax_model.msgpack
```

## 4. 运行模型测试

必须先完成词表和权重下载，再运行：

```bash
python scripts/test.py
```

测试脚本会依次执行：

1. 检查模型配置文件、词表和模型权重；
2. 检查 CUDA、DCU 或 CPU 是否可用；
3. 自动选择运行设备；
4. 加载 SapBERT Tokenizer；
5. 加载 SapBERT 模型配置；
6. 加载 `model.safetensors` 权重；
7. 对生物医学实体名称进行分词；
8. 提取最后一层的 `[CLS]` 实体向量；
9. 计算实体之间的余弦相似度；
10. 输出模型加载时间、推理时间和相似度矩阵。

测试脚本包含以下示例实体：

```text
myocardial infarction
heart attack
COVID-19
Coronavirus infection
high fever
Hydroxychloroquine
```

测试成功后，终端末尾会显示：

```text
模型测试完成：成功提取实体向量并计算语义相似度
```

## 5. 运行顺序

完整运行顺序为：

```bash
python scripts/download_weights.py
python scripts/test.py
```

不能直接跳过下载脚本运行测试，否则会提示缺少：

```text
config/vocab.txt
```

或：

```text
weight/model.safetensors
```

## 6. 提交代码前的清理

`config/vocab.txt` 和 `weight/model.safetensors` 仅用于本地模型运行，不提交到 Gitee 代码仓库。

完成测试后，提交代码前可以运行：

```bash
rm -f config/vocab.txt
rm -f weight/model.safetensors

rm -rf config/.cache
rm -rf weight/.cache
rm -rf scripts/__pycache__

touch weight/.gitkeep
```

清理后检查：

```bash
find . -maxdepth 3 -print | sort
```

提交前的目录结构应恢复为：

```text
SapBERT-from-PubMedBERT-fulltext/
├── config/
│   ├── config.json
│   ├── special_tokens_map.json
│   └── tokenizer_config.json
├── scripts/
│   ├── download_weights.py
│   └── test.py
├── weight/
│   └── .gitkeep
└── README.md
```

检查是否还存在需要排除的大文件：

```bash
find . -type f -size +100M -print
```

正常情况下不应有任何输出。

---

# Citation

```bibtex
@inproceedings{liu-etal-2021-self,
    title = "Self-Alignment Pretraining for Biomedical Entity Representations",
    author = "Liu, Fangyu and
      Shareghi, Ehsan and
      Meng, Zaiqiao and
      Basaldella, Marco and
      Collier, Nigel",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.334",
    pages = "4228--4238",
    abstract = "Despite the widespread success of self-supervised learning via masked language models, accurately capturing fine-grained semantic relationships in the biomedical domain remains a challenge. SapBERT introduces a self-alignment pretraining scheme for learning biomedical entity representations and achieves strong results on medical entity linking benchmarks.",
}
```