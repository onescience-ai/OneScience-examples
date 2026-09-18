<p align="center">
  <strong>
    <span style="font-size: 30px;">GPSite</span>
  </strong>
</p>

# 模型介绍

GPSite 是一个面向蛋白质结合位点预测的几何感知多任务网络，可同时预测蛋白质残基与 DNA、RNA、肽、蛋白质、ATP、血红素（HEM）以及多种金属离子的潜在结合位点。该方法利用预训练蛋白质语言模型生成的序列表征和预测结构完成结合位点预测，不依赖 MSA 或实验解析的蛋白质三维结构。

论文：

> **Genome-scale annotation of protein binding sites via language model and geometric deep learning**  
> https://doi.org/10.7554/eLife.93695

# 模型描述

GPSite 以蛋白质 FASTA 序列为输入。完整推理流程首先使用 ESMFold 预测蛋白质结构，并使用 ProtT5-XL-UniRef50 提取序列表征；随后结合预测结构与 DSSP 特征构建残基级几何表示，最后由 GPSite 图神经网络同时输出多种结合位点的逐残基预测分数。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 蛋白质结合位点预测 | 输入蛋白质序列，预测逐残基结合位点分数 |
| 多类型配体结合分析 | 同时预测 DNA、RNA、肽、蛋白质、ATP、HEM 和多种金属离子结合位点 |
| 无实验结构条件下的预测 | 使用 ESMFold 预测结构，无需预先提供实验解析结构 |
| 批量蛋白质序列分析 | 对 FASTA 中多条蛋白质序列统一执行结构预测、特征提取与位点预测 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU/DCU 运行 GPSite，完整流程中的 ESMFold 结构预测计算量和显存占用相对较高。
- GPSite 支持 CPU 运行，但不使用 GPU/DCU 时结构预测阶段会明显更慢。

### 安装运行环境

#### DCU环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

# 支持uv安装
pip install onescience[bio] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

#### 环境说明
- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考 `requirements.txt` 中声明的依赖版本，补充安装或调整相应依赖。
- ProtTrans 相关依赖建议固定为：

```bash
python -m pip install \
  "transformers==4.30.1" \
  "tokenizers==0.13.3" \
  "sentencepiece==0.1.99"
```

- ESMFold 相关依赖可安装为：

```bash
python -m pip install "fair-esm[esmfold]"
python -m pip install modelcif==0.7
```

- GPSite 还需要 `dllogger`。若在线安装遇到问题，可下载 `dllogger` 源码后，在源码目录本地安装：

```bash
python -m pip install /path/to/dllogger-master
```

### 权重与模型准备

GPSite 完整推理不仅依赖本仓库中的 GPSite 权重，还依赖 ProtT5、ESMFold 和 ESM-2 权重。首次使用前请完成以下准备。

#### 1）GPSite 模型权重

GPSite 仓库 `model/` 目录包含 5 个训练权重：

```text
model/
├── fold0.ckpt
├── fold1.ckpt
├── fold2.ckpt
├── fold3.ckpt
└── fold4.ckpt
```

推理时会依次加载上述 5 个模型，并对预测结果取平均。通过 ModelScope 下载的完整模型包应已包含 GPSite 推理所需的上述权重，正常情况下无需单独下载。

#### 2）ProtT5-XL-UniRef50

GPSite 使用 ProtT5-XL-UniRef50 提取蛋白质序列表征。模型下载地址为：

```text
https://zenodo.org/record/4644188
```

下载后需要保证 `scripts/run_infer.sh` 中的 `PROTTRANS_DIR` 指向实际模型目录。`scripts/predict.py` 会通过环境变量 `PROTTRANS_PATH` 读取该路径。

#### 3）ESMFold 与 ESM-2

GPSite 官方版本在首次运行时会自动下载 ESMFold 与 ESM-2。如果运行环境网络受限或离线环境建议提前准备本地权重。建议放置位置如下：

```text
weight/checkpoints/
├── esmfold_3B_v1.pt
├── esm2_t36_3B_UR50D.pt
└── esm2_t36_3B_UR50D-contact-regression.pt
```

同时需要保证 `scripts/run_infer.sh` 中的 `ESMFOLD_HUB_DIR` 指向项目内的 `weight` 目录。

其中，ESMFold v1 在内部还会加载 ESM-2，因此至少需要：

```text
esmfold_3B_v1.pt
esm2_t36_3B_UR50D.pt
```
建议同时保留：

```text
esm2_t36_3B_UR50D-contact-regression.pt
```

#### 4）DSSP

GPSite 在结构特征提取阶段会调用：

```text
scripts/feature_extraction/mkdssp
```

首次使用前请保证该文件具有执行权限：

```bash
chmod +x scripts/feature_extraction/mkdssp
```

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/GPSite --local_dir ./GPSite
cd GPSite
```
- GPSite 完整推理额外依赖 **ProtT5-XL-UniRef50、ESMFold、ESM-2 和 OpenFold**；请先按照“权重与模型准备”完成相关模型、权重以及 OpenFold 的安装。
- GPSite 额外依赖 OpenFold。OpenFold 需要安装到当前 Python/Conda 环境中，项目目录下不需要保留 OpenFold 源码；如需源码安装，可下载 OpenFold 源码并解压后，在源码目录执行：

```bash
python3 setup.py install
```

# 示例数据

GPSite 官方示例输入位于：

```text
conf/example/demo.fa
```

输入采用标准 FASTA 格式：

```text
>protein_id
MSEQUENCE...
```

GPSite 会根据 FASTA 文件名自动创建输出子目录。例如输入 `conf/example/demo.fa`，输出将位于：

```text
<OUTPUT_DIR>/demo/
```

# 推理

运行推理前，请先进入 GPSite 项目根目录，并确认 `scripts/run_infer.sh` 中的模型路径已经改为当前环境中的真实路径：

```bash
cd /path/to/GPSite
```

其中需要重点检查：

```bash
PROTTRANS_DIR="/path/to/prot_t5_xl_uniref50"
ESMFOLD_HUB_DIR="/path/to/weight"
```

推理脚本的基本用法为：

```bash
bash scripts/run_infer.sh <GPU_ID> <FASTA_PATH> <OUTPUT_DIR>
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `<GPU_ID>` | 使用的 GPU 编号，例如单卡环境通常为 `0` |
| `<FASTA_PATH>` | 输入 FASTA 文件路径，可使用相对路径或绝对路径 |
| `<OUTPUT_DIR>` | 输出根目录，脚本会根据 FASTA 文件名自动创建子目录 |

运行 demo：

```bash
bash scripts/run_infer.sh 0 ./conf/example/demo.fa ./results/
```

上述命令会读取 `conf/example/demo.fa`，并在 `./results/demo/` 下生成中间文件和最终预测结果。

运行自己的 FASTA：

```bash
bash scripts/run_infer.sh 0 /path/to/your.fa ./results/
```

例如输入文件为 `/public/home/user/test.fa`，输出根目录为 `./results/`，则最终预测结果通常位于：

```text
./results/test/pred/
```

推理过程中会依次完成 ESMFold 结构预测、ProtT5 序列表征提取、DSSP 结构特征提取和 GPSite 5 折模型预测。正常完成时，终端日志中应能看到类似信息：

```text
Feature extraction is done
Prediction is done
Results are saved in <OUTPUT_DIR>/<FASTA_NAME>/pred/
```

## 推理流程

完整推理流程为：

```text
FASTA 输入
  ↓
ESMFold 结构预测
  ↓
ProtT5 序列表征提取
  ↓
PDB / DSSP / 几何特征处理
  ↓
GPSite 5 折模型推理
  ↓
5 个模型输出取平均
  ↓
逐残基 10 类结合位点分数
```

# 输出说明

假设输入文件为 `demo.fa`，输出根目录为 `./results/`，则最终预测结果位于：

```text
./results/demo/pred/
```

主要结果包括：

```text
pred/
├── overview.txt
├── A0A009IHW8.txt
└── A0A011QK89.txt
```

其中：

- `overview.txt`：汇总所有输入蛋白质的整体预测结果；
- `<Protein_ID>.txt`：对应蛋白质的逐残基预测结果。

逐残基预测文件包含以下 10 类结合位点分数：

```text
DNA
RNA
Peptide
Protein
ATP
HEM
ZN
CA
MG
MN
```

按照 GPSite 官方说明，归一化预测分数大于 `0.5` 的残基可视为预测 binding site。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |


# 引用与许可证

- GPSite 原始论文：[Genome-scale annotation of protein binding sites via language model and geometric deep learning](https://doi.org/10.7554/eLife.93695)。
- GPSite 官方源码采用 MIT License，详见仓库根目录 `LICENSE`。
- 如果在科研工作中使用本仓库，建议引用 GPSite 原始论文及 OneScience 相关项目信息；如实际使用 ESMFold、ProtT5 等外部模型，还应按照对应项目要求补充引用。
