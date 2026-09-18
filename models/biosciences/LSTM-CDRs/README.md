<p align="center">
  <strong>
    <span style="font-size: 30px;">LSTM_CDRs</span>
  </strong>
</p>

# 模型介绍

LSTM_CDRs 是一个基于长短期记忆网络（Long Short-Term Memory, LSTM）的 CDR 氨基酸序列生成模型。给定一组 CDR 序列，模型可以学习训练集中序列的分布，并在训练完成后通过采样生成新的 CDR 序列。

# 模型描述

该项目使用循环神经网络对氨基酸序列进行自回归建模。输入序列会先经过 padding 和 one-hot 编码，随后送入多层 LSTM 或 GRU 网络进行训练。训练完成后，脚本可加载指定 epoch 的模型权重，从学习到的序列分布中采样生成新的 CDR 序列。

官方代码改编自 `LSTM_peptides`，用于 VHH CDR 序列设计相关任务。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| CDR 序列生成 | 基于给定 CDR 训练集学习氨基酸序列分布，并采样生成新的候选序列。 |
| 本地 LSTM/GRU 训练 | 用户通过本仓库脚本在本地、GPU 或 DCU 平台训练 LSTM/GRU 序列生成模型。 |
| 权重加载和采样复现 | 加载训练产生的 checkpoint 权重，按指定长度、温度和采样数量生成 CDR 序列。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行训练和采样任务。
- CPU 可用于小规模连通性验证，但完整训练和大规模采样速度较慢。
- DCU 用户需要使用与当前集群匹配的 DTK、TensorFlow 和 OneScience 环境。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/LSTM_CDRs --local_dir ./LSTM_CDRs
cd LSTM_CDRs
```

### 安装运行环境

#### DCU 环境

```bash
# 请首先激活 DTK 及 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

#### 环境说明
- 实际运行时如遇依赖缺失或版本兼容问题，请参考 requirements.txt 和 environment.yml 中声明的依赖版本，按需补充安装或调整环境。
- 在运行过程中如果遇到tensorflow相关问题，可以将本环境中的tensorflow卸载，按如下方式解决：
```bash
1. 在平台下载tensorflow
wget --content-disposition 'https://download.sourcefind.cn:65024/file/4/tensorflow/DAS1.8/tensorflow-2.13.1+das.opt1.dtk2604-cp311-cp311-manylinux_2_28_x86_64.whl'

2. 安装tensorflow
pip install tensorflow* 

3. 加载对应版本dtk
module load compiler/dtk/26.04 
```
- 用户也可根据自己的实际平台根据requirements.txt和environment.yml声明的依赖自行搭建运行环境。

## 快速验证

```bash
python LSTM_CDRs.py --help
ls data
```

训练数据应包含：

```text
data/Cluster1.csv
data/Cluster2.csv
data/Cluster3.csv
data/Cluster4.csv
```

各数据文件信息如下：

| 数据文件 | 序列数量 | 序列长度 |
| --- | ---: | --- |
| `data/Cluster1.csv` | 2629 | 36 |
| `data/Cluster2.csv` | 4146 | 36 |
| `data/Cluster3.csv` | 2990 | 35 |
| `data/Cluster4.csv` | 11952 | 36 |

## 权重与数据准备

当前仓库已经包含训练数据：

```text
data/Cluster1.csv
data/Cluster2.csv
data/Cluster3.csv
data/Cluster4.csv
```

仓库未提供预训练权重文件。采样任务所需的权重需要先通过训练生成。

训练完成后，每个实验目录下会生成：

```text
<run_name>/
  flags.txt
  <run_name>_loss_plot.pdf
  sampled_sequences_temp1.25.csv
  checkpoint/
    model.json
    model.p
    model.hdf5
    model_epoch_0.hdf5
    model_epoch_1.hdf5
    ...
```

采样或微调时，`--modfile` 应指向某个已存在的 epoch 权重，例如：

```text
Cluster1_LSTM/checkpoint/model_epoch_100.hdf5
```

同时，同一个 `checkpoint/` 目录中需要保留：

```text
model.p
model.hdf5
model_epoch_*.hdf5
```

## 训练

### 使用脚本运行最小训练验证

```bash
python LSTM_CDRs.py \
  --name smoke_Cluster1 \
  --dataset data/Cluster1.csv \
  --layers 1 \
  --neurons 16 \
  --epochs 1 \
  --batch_size 64 \
  --dropout 0.1 \
  --sample 10
```

该命令用于确认数据读取、padding、one-hot 编码、模型训练和采样流程是否连通。

查看输出：

```bash
ls smoke_Cluster1
ls smoke_Cluster1/checkpoint
head smoke_Cluster1/sampled_sequences_temp1.25.csv
```

### 训练 Cluster1 模型

```bash
python LSTM_CDRs.py \
  --name Cluster1_LSTM \
  --dataset data/Cluster1.csv \
  --layers 2 \
  --neurons 64 \
  --epochs 200 \
  --dropout 0.2
```

该命令默认会在训练后采样 100 条序列，并保存到：

```text
Cluster1_LSTM/sampled_sequences_temp1.25.csv
```

查看训练日志和权重：

```bash
ls Cluster1_LSTM
ls Cluster1_LSTM/checkpoint
```

### 训练四个 Cluster

```bash
python LSTM_CDRs.py --name Cluster1_LSTM --dataset data/Cluster1.csv --layers 2 --neurons 64 --epochs 200 --dropout 0.2
python LSTM_CDRs.py --name Cluster2_LSTM --dataset data/Cluster2.csv --layers 2 --neurons 64 --epochs 200 --dropout 0.2
python LSTM_CDRs.py --name Cluster3_LSTM --dataset data/Cluster3.csv --layers 2 --neurons 64 --epochs 200 --dropout 0.2
python LSTM_CDRs.py --name Cluster4_LSTM --dataset data/Cluster4.csv --layers 2 --neurons 64 --epochs 200 --dropout 0.2
```

## 采样

采样是训练完成后的生成步骤。脚本会加载已有模型权重，并从模型学习到的 CDR 序列分布中生成新序列。

### Cluster1 采样

```bash
python LSTM_CDRs.py \
  --name Cluster1_LSTM \
  --dataset data/Cluster1.csv \
  --modfile Cluster1_LSTM/checkpoint/model_epoch_100.hdf5 \
  --train False \
  --sample 10000 \
  -f 36 \
  -m 36
```

该命令默认使用：

```text
训练数据：data/Cluster1.csv
模型权重：Cluster1_LSTM/checkpoint/model_epoch_100.hdf5
采样数量：10000
最短长度：36
最长长度：36
输出文件：Cluster1_LSTM/sampled_sequences_temp1.25.csv
```

### Cluster3 采样

Cluster3 原始序列长度为 35，因此采样时建议使用 `-f 35 -m 35`：

```bash
python LSTM_CDRs.py \
  --name Cluster3_LSTM \
  --dataset data/Cluster3.csv \
  --modfile Cluster3_LSTM/checkpoint/model_epoch_100.hdf5 \
  --train False \
  --sample 10000 \
  -f 35 \
  -m 35
```

### 查看采样结果

```bash
wc -l Cluster1_LSTM/sampled_sequences_temp1.25.csv
head Cluster1_LSTM/sampled_sequences_temp1.25.csv
```

检查生成序列长度：

```bash
awk '{print length($0)}' Cluster1_LSTM/sampled_sequences_temp1.25.csv | sort -n | uniq -c
```

检查生成序列与训练集重复数量：

```bash
grep -Fxf data/Cluster1.csv Cluster1_LSTM/sampled_sequences_temp1.25.csv | wc -l
```

### 微调

```bash
python LSTM_CDRs.py \
  --name Cluster1_to_Cluster2_finetune \
  --dataset data/Cluster2.csv \
  --modfile Cluster1_LSTM/checkpoint/model_epoch_100.hdf5 \
  --train False \
  --finetune True \
  --epochs 50 \
  --layers 2 \
  --neurons 64 \
  --dropout 0.2
```

### 交叉验证

```bash
python LSTM_CDRs.py \
  --name Cluster1_CV \
  --dataset data/Cluster1.csv \
  --layers 2 \
  --neurons 64 \
  --epochs 50 \
  --dropout 0.2 \
  --cv 5
```
## 常用参数

### 训练参数

| 参数 | 说明 | 默认/示例 |
| --- | --- | --- |
| `--dataset` | 训练数据 CSV 文件路径 | `data/Cluster1.csv` |
| `--name` | 实验名称，也是输出目录名 | `Cluster1_LSTM` |
| `--layers` | LSTM/GRU 层数 | 示例 `2` |
| `--neurons` | 每层神经元数量 | 示例 `64` |
| `--epochs` | 训练轮数 | 示例 `200` |
| `--batch_size` | batch size | 默认 `128` |
| `--dropout` | dropout 比例；第 n 层使用 `n * dropout` | 示例 `0.2` |
| `--cell` | 循环神经网络单元类型 | `LSTM` 或 `GRU` |
| `--lr` | Adam 学习率 | 默认 `0.01` |
| `--valsplit` | 验证集比例 | 默认 `0.2` |
| `--cv` | 交叉验证折数 | 默认不启用 |

### 采样参数

| 参数 | 说明 | 默认/示例 |
| --- | --- | --- |
| `--train False` | 不训练，加载已有模型进行采样 | 采样时必须设置 |
| `--modfile` | 已训练 epoch 权重路径 | `Cluster1_LSTM/checkpoint/model_epoch_100.hdf5` |
| `--sample` | 采样生成序列数量 | 示例 `10000` |
| `--temp` | 采样温度 | 默认 `1.25` |
| `-f`, `--fminlen` | 生成序列最短长度 | Cluster1/2/4 用 `36`，Cluster3 用 `35` |
| `-m`, `--maxlen` | 生成序列最长长度 | Cluster1/2/4 用 `36`，Cluster3 用 `35` |
| `--startchar` | 采样起始字符 | 默认 `B` |

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 引用与许可证

- 相关工作：A. T. Mueller, J. A. Hiss, G. Schneider, "Recurrent Neural Network Model for Constructive Peptide Design", Journal of Chemical Information and Modeling, 2018, DOI: 10.1021/acs.jcim.7b00414。
- 应用论文：P. Arras et al., "AI/ML combined with Next Generation Sequencing of VHH immune repertoires enables the rapid identification of de novo humanized and sequence-optimized single domain antibodies: a prospective case study", Frontiers in Molecular Biosciences, 2023, DOI: 10.3389/fmolb.2023.1249247。
- 本项目使用 MIT License，详见仓库根目录 `LICENSE`。数据和模型权重的具体使用条款请以对应发布方说明为准。
