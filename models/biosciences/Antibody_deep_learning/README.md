<p align="center">
  <strong>
    <span style="font-size: 30px;">Antibody Deep Learning</span>
  </strong>
</p>

# 模型介绍

Antibody Deep Learning 是一个面向抗体 CDR3 序列分析的深度学习复现项目，主要包含两个任务：

1. 使用卷积神经网络 CNN 预测 CTLA-4 和 PD-1 抗体序列是否为 binder。
2. 使用生成对抗网络 GAN 生成面向 CTLA-4 和 PD-1 的合成 CDR3K/CDR3H 序列。

原始项目以 RMarkdown 为主入口，通过 R `keras` / `reticulate` 调用 Python TensorFlow 后端。本仓库保留官方数据、官方已训练权重和原始说明，同时在 `scripts/` 目录中提供适配当前 TensorFlow/DCU 环境的等价运行脚本。

论文：

Predicting antibody binders and generating synthetic antibodies using deep learning

https://doi.org/10.1080/19420862.2022.2069075

# 模型描述

本项目包含两类模型。

| 模型 | 任务 | 输入 | 输出 |
| --- | --- | --- | --- |
| CNN | 判断 CTLA-4 / PD-1 抗体序列是否为 binder | CDR3K + CDR3H，经 padding 和 BLOSUM62 编码后为 `36 x 22 x 1` | 二分类概率，non-binder / binder |
| GAN | 生成 CDR3 序列 | 100 维随机噪声 | `32 x 22 x 1` 的氨基酸图像，再解码为 CDR3 序列 |

CNN 分别训练两个模型：

| 模型路径 | 靶点 | 说明 |
| --- | --- | --- |
| `weight/CNN/model_c1` | CTLA-4 | 官方已训练 CNN SavedModel |
| `weight/CNN/model_p1` | PD-1 | 官方已训练 CNN SavedModel |

GAN 共 15 个生成器，对应不同 target / chain / V gene 组合：

| 编号 | 官方权重路径 | 分组 |
| --- | --- | --- |
| 1 | `weight/GAN/GAN_model_1` | CTLA4 heavy IGHV3-33*01 |
| 2 | `weight/GAN/GAN_model_2` | CTLA4 heavy IGHV1-18*04 |
| 3 | `weight/GAN/GAN_model_3` | CTLA4 heavy IGHV3-20*01 |
| 4 | `weight/GAN/GAN_model_4` | CTLA4 heavy IGHV4-39*01 |
| 5 | `weight/GAN/GAN_model_5` | CTLA4 light IGKV3-20*01 |
| 6 | `weight/GAN/GAN_model_6` | CTLA4 light IGKV1D-39*01 |
| 7 | `weight/GAN/GAN_model_7` | CTLA4 light IGKV1-17*01 |
| 8 | `weight/GAN/GAN_model_8` | CTLA4 light IGKV1-16*01 |
| 9 | `weight/GAN/GAN_model_9` | PD1 heavy IGHV4-4*07 |
| 10 | `weight/GAN/GAN_model_10` | PD1 heavy IGHV3-33*03 |
| 11 | `weight/GAN/GAN_model_11` | PD1 heavy IGHV1-18*04 |
| 12 | `weight/GAN/GAN_model_12` | PD1 light IGKV1-17*01 |
| 13 | `weight/GAN/GAN_model_13` | PD1 light IGKV1-6*02 |
| 14 | `weight/GAN/GAN_model_14` | PD1 light IGKV3-15*01 |
| 15 | `weight/GAN/GAN_model_15` | PD1 light IGKV1-9*01 |

# 适用场景

| 场景 | 说明 |
| --- | --- |
| CTLA-4 / PD-1 binder 分类 | 使用仓库内置 CNN 模型，对 CDR3K + CDR3H 序列进行 BLOSUM62 编码后预测 binder / non-binder，可复现论文中的抗体结合分类任务。 |
| 合成抗体 CDR3 序列生成 | 使用 15 个 GAN generator，按 CTLA-4 / PD-1、heavy / light chain 及 V gene 分组生成合成 CDR3 序列。 |
| 抗体工程方法复现 | 复现论文中将抗体 CDR3 序列转为二维“抗体图像”、训练 CNN 分类器、使用 GAN 学习序列分布的核心流程。 |
| 解释性分析和序列优化参考 | 结合原始 RMarkdown 中的模型评估、ROC 分析和 in silico mutagenesis 思路，分析影响 binder 分类的重要 CDR3 位点。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

### 硬件要求

- CPU 可用于数据预处理、小规模推理和连通性验证。
- 推荐使用 GPU/DCU 进行训练和批量推理。
- DCU 用户需要加载与当前集群匹配的 DTK 模块，并先验证 TensorFlow 基础算子可正常运行。

## 安装运行环境

### DCU 环境

```bash
# 请首先激活 DTK 及 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 环境说明

- 搭建好 OneScience 基础环境后，用户还需额外准备 R 运行环境和 R 包。安装示例：

```bash
module load R/3.6.3-gcc-7.3.1
mkdir -p ~/R/library/3.6 ~/tmp
export R_LIBS_USER=$HOME/R/library/3.6
```

如果集群中的 R 模块路径不是 `/public/software/apps/R-3.6.3/bin`，请先用下面命令确认实际路径，并同步替换后续命令中的 `PATH`：

```bash
which Rscript
Rscript --version
```
由于 R 3.6.3 版本较老，部分 CRAN 最新包不再兼容，建议使用 CRAN 历史快照安装依赖：
```bash
env -i \
HOME=$HOME \
USER=$USER \
PATH=/usr/bin:/bin:/public/software/apps/R-3.6.3/bin \
R_LIBS_USER=$HOME/R/library/3.6 \
TMPDIR=$HOME/tmp \
Rscript -e 'options(repos=c(CRAN="https://packagemanager.posit.co/cran/2023-10-20")); install.packages(c("reticulate","dplyr","ggplot2","readr","tidyr","purrr","tibble","stringr","forcats","mltools","caret","pROC","remotes"), type="source")'
```
安装完成后验证 R 包可正常加载：

```bash
env -i \
HOME=$HOME \
USER=$USER \
PATH=/usr/bin:/bin:/public/software/apps/R-3.6.3/bin \
R_LIBS_USER=$HOME/R/library/3.6 \
TMPDIR=$HOME/tmp \
Rscript -e 'library(reticulate); library(caret); library(pROC); cat("R packages OK\n")'
```
后续运行 R 脚本时仍需显式传入 `R_LIBS_USER=$HOME/R/library/3.6`，否则可能出现 `there is no package called ...` 的报错。

- 运行过程中遇到 TensorFlow 相关问题，可以使用平台提供的适配版 TensorFlow wheel，并加载匹配的 DTK 模块。例如：

```bash
# 1. 下载平台 TensorFlow wheel
wget --content-disposition 'https://download.sourcefind.cn:65024/file/4/tensorflow/DAS1.8/tensorflow-2.13.1+das.opt1.dtk2604-cp311-cp311-manylinux_2_28_x86_64.whl'

# 2. 安装 TensorFlow
pip install tensorflow*

# 3. 加载对应 DTK
module load compiler/dtk/26.04
```

# 快速开始

## 1. 下载模型包

```bash
modelscope download --model OneScience/Antibody_deep_learning --local_dir ./Antibody_deep_learning
cd Antibody_deep_learning
```

# 数据和权重说明

## 内置数据

| 路径 | 说明 |
| --- | --- |
| `model/CNN/all_ab_pre_post.txt` | CNN 原始输入表，包含 CDR3K、CDR3H、antigen、pre/post frequency、fold change 等信息。 |
| `model/BLOSUM62_with_deletion.Rdata` | BLOSUM62 编码矩阵，包含 20 种氨基酸、`X` 和 gap `-`。 |
| `model/CNN/c1.RDS` / `model/CNN/p1.RDS` | CTLA-4 / PD-1 的 train/test 划分对象。 |
| `model/CNN/*train*.RDS` / `model/CNN/*test*.RDS` | CNN 训练和测试张量及 one-hot 标签。 |
| `model/GAN/seq_all.RDS` | GAN 预处理 CDR3 序列，按 target/chain/V gene 分组。 |
| `model/GAN/seq_all_encoded.RDS` | GAN BLOSUM62 编码后的训练张量列表。 |

## 内置权重

| 路径 | 说明 |
| --- | --- |
| `weight/CNN/model_c1` | 官方 CTLA-4 CNN SavedModel。 |
| `weight/CNN/model_p1` | 官方 PD-1 CNN SavedModel。 |
| `weight/GAN/GAN_model_1` 到 `weight/GAN/GAN_model_15` | 官方 15 个 GAN generator SavedModel。 |

# 推理示例

## 1. CNN 模型推理

用途：加载 `weight/CNN/model_c1` 和 `weight/CNN/model_p1`，实现 CTLA-4 / PD-1 binder 分类。

```bash
env -i \
HOME=$HOME \
USER=$USER \
PATH=$PATH:/public/software/apps/R-3.6.3/bin \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
R_LIBS_USER=$HOME/R/library/3.6 \
RETICULATE_PYTHON=$(which python) \
PYTHONNOUSERSITE=1 \
TMPDIR=$HOME/tmp \
Rscript scripts/02_cnn_inference.R
```

输出文件：

```text
model/CNN/c1_tf218_inference_result.RDS
model/CNN/p1_tf218_inference_result.RDS
```

## 2. GAN 模型推理

用途：加载 `weight/GAN/GAN_model_1` 到 `weight/GAN/GAN_model_15`，每个模型生成 100 条 CDR3 序列。

```bash
env -i \
HOME=$HOME \
USER=$USER \
PATH=$PATH:/public/software/apps/R-3.6.3/bin \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
R_LIBS_USER=$HOME/R/library/3.6 \
RETICULATE_PYTHON=$(which python) \
PYTHONNOUSERSITE=1 \
TMPDIR=$HOME/tmp \
Rscript scripts/03_gan_inference.R
```

输出文件：

```text
model/GAN/gen_seq_tf218.RDS
model/GAN/gen_seq_tf218.tsv
```

# 训练示例

## 1. 数据预处理

用途：生成 CNN/GAN 中间训练数据。

```bash
env -i \
HOME=$HOME \
USER=$USER \
PATH=$PATH:/public/software/apps/R-3.6.3/bin \
R_LIBS_USER=$HOME/R/library/3.6 \
TMPDIR=$HOME/tmp \
Rscript scripts/01_prepare_data_compat.R
```

输出包括：

```text
model/CNN/c1_train.RDS
model/CNN/c1_test.RDS
model/CNN/p1_train.RDS
model/CNN/p1_test.RDS
model/GAN/seq_all_encoded.RDS
```

## 2. CNN 训练

先导出 Python 可读数据：

```bash
env -i \
HOME=$HOME \
USER=$USER \
PATH=$PATH:/public/software/apps/R-3.6.3/bin \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
R_LIBS_USER=$HOME/R/library/3.6 \
RETICULATE_PYTHON=$(which python) \
PYTHONNOUSERSITE=1 \
TMPDIR=$HOME/tmp \
Rscript scripts/04_export_cnn_npz.R
```

训练：

```bash
python scripts/05_train_cnn.py
```

输出：

```text
weight/CNN/model_c1_dcu
weight/CNN/model_p1_dcu
weight/CNN/model_c1_dcu_eval.npz
weight/CNN/model_p1_dcu_eval.npz
```

## 3. GAN 训练

先导出 Python 可读数据：

```bash
env -i \
HOME=$HOME \
USER=$USER \
PATH=$PATH:/public/software/apps/R-3.6.3/bin \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
R_LIBS_USER=$HOME/R/library/3.6 \
RETICULATE_PYTHON=$(which python) \
PYTHONNOUSERSITE=1 \
TMPDIR=$HOME/tmp \
Rscript scripts/06_export_gan_npz.R
```

单模型 smoke test：

```bash
python scripts/07_train_gan.py --model-id 1 --rounds 20
```

单模型完整训练：

```bash
python scripts/07_train_gan.py --model-id 1 --rounds 100
```

全量 15 个模型训练：

```bash
for i in $(seq 1 15); do
  echo "===== training GAN model $i ====="
  python scripts/07_train_gan.py --model-id $i --rounds 100
done
```

输出：

```text
weight/GAN/GAN_model_1_dcu 到 weight/GAN/GAN_model_15_dcu
weight/GAN/GAN_model_1_dcu_loss.npz 到 weight/GAN/GAN_model_15_dcu_loss.npz
```

## 4. 使用新训练 GAN 模型生成序列

单模型：

```bash
python scripts/08_generate_from_trained_gan.py \
  --model-id 1 \
  --n-seq 100 \
  --out-tsv model/GAN/gen_seq_trained_model_1_dcu.tsv
```

全量：

```bash
python scripts/08_generate_from_trained_gan.py \
  --model-id 0 \
  --n-seq 100 \
  --out-tsv model/GAN/gen_seq_trained_all_dcu.tsv
```

生成统计：

```bash
python - <<'PY'
import pandas as pd
import re

df = pd.read_csv("model/GAN/gen_seq_trained_all_dcu.tsv", sep="\t")
df["length"] = df["aa"].astype(str).str.len()
df["valid"] = df["aa"].astype(str).str.fullmatch(r"[ARNDCQEGHILKMFPSTWYV]+")

summary = (
    df.groupby(["model_id", "group"])
      .agg(
          n_seq=("aa", "size"),
          n_unique=("aa", "nunique"),
          min_len=("length", "min"),
          median_len=("length", "median"),
          max_len=("length", "max"),
          valid_rate=("valid", "mean"),
      )
      .reset_index()
)

print(summary.to_string(index=False))
summary.to_csv("model/GAN/gen_seq_trained_all_dcu_summary.tsv", sep="\t", index=False)
PY
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Antibody Deep Learning 原始论文：[Predicting antibody binders and generating synthetic antibodies using deep learning](https://doi.org/10.1080/19420862.2022.2069075)。

- 论文信息：Yoong Wearn Lim, Adam S. Adler, David S. Johnson. *mAbs* 14(1):2069075, 2022. DOI: [10.1080/19420862.2022.2069075](https://doi.org/10.1080/19420862.2022.2069075)。

- 原始代码和数据来源：[ywlim/Antibody_deep_learning](https://github.com/ywlim/Antibody_deep_learning)。论文数据可用性说明中给出了该仓库地址。

- 相关源码使用 Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International(CC BY-NC-SA 4.0)，详见仓库根目录 `LICENSE`。使用、修改和再发布本项目内容时，请遵循署名、非商业使用和相同方式共享等许可要求。

- 如果在科研工作中使用本项目，建议同时引用原论文以及 OneScience 相关信息。

