<p align="center">
  <strong>
    <span style="font-size: 30px;">UTRGAN</span>
  </strong>
</p>

# 模型介绍

UTRGAN 是面向 5′ UTR 设计的生成与优化模型，可生成候选 5′ UTR 序列，并对基因表达、平均核糖体负载（Mean Ribosome Load，MRL）和翻译效率（Translation Efficiency，TE）进行预测与排序。

模型包已经包含基础运行所需的数据和预训练权重，无需在推理时额外下载模型文件。



# 模型描述

UTRGAN 由多个相互配合的模型组成：

- WGAN-GP Generator：从 40 维随机向量生成最长 128 nt 的候选 5′ UTR；
- WGAN-GP Critic：参与生成模型训练；
- Xpresso：预测与基因表达相关的分数；
- FramePool：预测 MRL；
- MTtrans：预测 TE；
- G4Boost：用于 G4 相关分类和回归分析。

WGAN、Xpresso 和 FramePool 使用 TensorFlow/Keras，MTtrans 使用 PyTorch。为兼容官方发布的旧版 H5 文件，运行时使用 Legacy Keras 加载模型。



# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 5′ UTR 候选生成 | 使用官方预训练 WGAN 批量生成 5′ UTR 序列 |
| MRL 预测与排序 | 使用 FramePool 计算 MRL 分数并筛选候选序列 |
| TE 预测与排序 | 使用 MTtrans 计算 TE 分数并筛选候选序列 |
| MRL/TE 定向优化 | 冻结预训练模型，通过更新 latent noise 优化目标分数 |
| 基因表达优化 | 结合 Xpresso 对候选序列进行表达相关预测 |
| WGAN-GP 训练 | 使用仓库自带 UTRdb2 数据验证或重新训练生成模型 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 支持 CPU 推理；
- 推荐使用 OneScience 支持的 DCU 环境进行批量生成、排序和训练；
- TensorFlow 与 PyTorch 通过兼容接口访问同一张 DCU；
- 完整训练所需时间与候选数量、batch size 和设备性能有关。

### 下载模型包

安装 ModelScope 命令行工具后下载模型包：

```bash
pip install modelscope
modelscope download --model OneScience/UTRGAN --local_dir ./UTRGAN
cd UTRGAN
```

### 安装运行环境

创建并激活 Python 3.11 环境，然后安装 OneScience 生物科学基础环境：

```bash
conda create -n utrgan python=3.11 -y
conda activate utrgan
pip install onescience[bio-dcu]
```

再安装 `requirements.txt` 中相对 OneScience 新增或需要替换的依赖：

```bash
python -m pip install --no-deps -r requirements.txt
```

> 提示：本模型所使用dtk为2604版本，相应的tensorflow也随之升级到2.18版本。



检查框架和设备：

```bash
python - <<'PY'
import tensorflow as tf
import torch

print("TensorFlow:", tf.__version__)
print("TensorFlow devices:", tf.config.list_physical_devices("GPU"))
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("DCU available:", torch.cuda.is_available())
PY
```

### 权重与数据准备

模型包已包含基础生成、预测、排序和训练验证所需的资源：

| 资源 | 位置 | 用途 |
| --- | --- | --- |
| UTRdb2 | `conf/data/utrdb2.csv` | WGAN-GP 训练数据 |
| motif 数据 | `conf/data/motifs.csv` | motif 统计与优化分析 |
| WGAN Generator | `weight/checkpoint_3000.h5` | 生成候选 5′ UTR |
| FramePool | `weight/utr_model_combined_residual_new.h5` | MRL 预测 |
| Xpresso | `weight/humanMedian_trainepoch.11-0.426.h5` | 表达相关预测 |
| Xpresso | `weight/GM12878_trainepoch.06-0.5062.h5` | GM12878 表达相关预测 |
| Xpresso | `weight/K562_trainepoch.11-0.4917.h5` | K562 表达相关预测 |
| MTtrans | `weight/mttrans/RL_hard_share_MTL/3R/schedule_MTL-model_best_cv1.pth` | TE 预测 |
| G4Boost | `weight/G4Boost_classifier.json` | G4 分类 |
| G4Boost | `weight/G4Boost_regressor.json` | G4 回归 |

按官方训练脚本过滤并去重后，UTRdb2 可得到约 33,250 条长度为 65–128 nt 的序列。基础运行不需要额外下载数据或预计算特征。

### 可选依赖

`requirements.txt` 已列出 G4 和分析脚本使用的 XGBoost、ViennaRNA、logomaker、ruptures 与 cliffs-delta。

NUPACK 仅用于可选的 MFE 预处理脚本。由于 NUPACK 4 需要按照其发布方的许可和安装方式单独获取，本模型包不通过普通 PyPI 依赖自动安装它。基础推理、MRL/TE 排序和 WGAN-GP 训练均不需要 NUPACK。

### 快速推理

使用官方预训练 WGAN 生成候选序列，并分别按照 MRL 和 TE 排序：

**用途：** 在 DCU 上批量生成 5′ UTR 候选，并输出 MRL 和 TE 两套排序结果。

```bash
python scripts/predict.py \
  --device dcu \
  --device-id 0 \
  --num-candidates 1024 \
  --batch-size 128 \
  --seed 33 \
  --output-dir outputs/pretrained_batch_ranking
```

结果保存在：

```text
outputs/pretrained_batch_ranking/
├── all_candidates_scores.csv
├── ranked_by_mrl.csv
├── ranked_by_te.csv
├── generator_probabilities.npy
└── summary.json
```

其中：

- `all_candidates_scores.csv` 保存全部候选和两项预测分数；
- `ranked_by_mrl.csv` 按 MRL 从高到低排序；
- `ranked_by_te.csv` 按 TE 从高到低排序；
- `is_duplicate` 标记重复序列；
- MRL 和 TE 的量纲不同，不应直接相加原始分数。



### MRL 定向优化

**用途：** 冻结 WGAN 和 FramePool，通过更新 latent noise 提升候选序列的 MRL 分数。

```bash
python scripts/optimize_te_mrl.py \
  -gpu 0 \
  -task mrl \
  -bs 64 \
  -s 10 \
  --output-dir outputs/optimization_mrl
```

该流程冻结 WGAN 和 FramePool 权重，仅更新 latent noise。

### TE 定向优化

**用途：** 冻结 WGAN 和 MTtrans，通过更新 latent noise 提升候选序列的 TE 分数。

```bash
python scripts/optimize_te_mrl.py \
  -gpu 0 \
  -task te \
  -bs 64 \
  -s 10 \
  --output-dir outputs/optimization_te
```

该流程冻结 WGAN 和 MTtrans 权重，仅更新 latent noise，因此不属于预训练模型微调。

### 训练

UTRGAN 官方支持 WGAN-GP 训练。被训练的是 Generator 和 Critic，Xpresso、FramePool 和 MTtrans 不参与该训练入口。

使用仓库自带数据执行完整数据流程 1 epoch：



```bash
python scripts/train.py \
  -gpu 0 \
  -bs 64 \
  -lr 5 \
  -mxl 128 \
  -dim 40 \
  --epochs 1 \
  --output-dir outputs/train_1epoch
```

`-lr 5` 按上游 README 的定义表示学习率 `1e-5`。适配后的入口增加了 `--epochs` 参数，并修正了上游脚本中与该定义不一致的学习率表达式。

执行上游默认规模的完整训练：

使用完整 UTRdb2 训练集运行 4000 epochs，重新训练 WGAN-GP Generator 和 Critic。

```bash
python scripts/train.py \
  -gpu 0 \
  -bs 64 \
  -lr 5 \
  -mxl 128 \
  -dim 40 \
  --epochs 4000 \
  --output-dir outputs/train_full
```

完整训练耗时较长，应根据设备资源和训练日志决定是否持续运行。训练生成的 checkpoint 保存在指定输出目录中，不会覆盖 `weight/` 下的官方预训练权重。



# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | [OneScience](https://gitee.com/onescience-ai/onescience) | [OneSkills](https://gitee.com/onescience-ai/oneskills) |
| GitHub | [OneScience](https://github.com/onescience-ai/OneScience) | [OneSkills](https://github.com/onescience-ai/oneskills) |

# 引用与许可证

- 上游实现：[ciceklab/UTRGAN](https://github.com/ciceklab/UTRGAN)
- Xpresso：[vagarwal87/Xpresso](https://github.com/vagarwal87/Xpresso)
- FramePool：[Karollus/5UTR](https://github.com/Karollus/5UTR)
- MTtrans：[holab-hku/MTtrans](https://github.com/holab-hku/MTtrans)

- UTRGAN 上游项目采用 [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/) 许可，限学术和非商业用途；商业使用请联系上游作者。各第三方模型、数据和软件分别受其原始许可证及使用条款约束。
- 本仓库的 DCU 适配继续采用相同许可证。

