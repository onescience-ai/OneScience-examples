<p align="center">
  <strong>
    <span style="font-size: 30px;">TemStaPro</span>
  </strong>
</p>

# 模型介绍

TemStaPro（Temperatures of Stability for Proteins）是一个基于蛋白质语言模型表征的蛋白质热稳定性预测工具。该方法以蛋白质 FASTA 序列为输入，使用 ProtTrans/ProtT5 生成序列表征，再通过多个温度阈值对应的分类器预测蛋白质在不同温度范围下的稳定性。

论文：

> **TemStaPro: protein thermostability prediction using sequence representations from protein language models**  
> https://doi.org/10.1093/bioinformatics/btae157

# 模型描述

TemStaPro 使用 ProtT5-XL-Half-UniRef50 编码蛋白质序列，并基于生成的 mean embedding 或逐残基 embedding 完成热稳定性预测。默认模式使用多个二分类器分别判断蛋白质在 40、45、50、55、60 和 65 °C 阈值下的稳定性，并综合这些分类结果给出预测温度区间。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 蛋白质热稳定性预测 | 根据蛋白质序列预测稳定温度区间 |
| 多温度阈值分类 | 分别评估蛋白质在 40–65 °C 等阈值下的稳定性 |
| 逐残基稳定性分析 | 输出蛋白质各氨基酸位置的局部预测结果 |
| 局部片段稳定性分析 | 基于滑动窗口预测蛋白质不同区域的热稳定性 |
| 蛋白质工程与筛选 | 辅助筛选潜在耐热蛋白质或候选突变体 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- TemStaPro 支持 CPU 和 GPU 运行。
- 主要计算开销来自 ProtT5 embedding 生成，推荐使用 GPU/DCU 加速。
- 官方测试中，1000 条平均长度约 1137 aa 的蛋白质序列在普通笔记本 CPU 上约需 10 小时，而 RTX 2080 Ti GPU 系统约需 10 分钟，因此批量预测时建议使用加速卡。

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

- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考 environment_CPU.yml 或 environment_GPU.yml 中声明的依赖版本，补充安装或调整相应依赖。
### 权重与模型准备

- TemStaPro 推理需要两部分模型资源：

(1) TemStaPro 分类器权重；
(2) ProtT5-XL-Half-UniRef50 预训练模型。

-  TemStaPro 推理时不需要额外下载数据集，常规使用流程直接以用户自己的FASTA 文件作为输入。

#### 1）TemStaPro 分类器权重

当前仓库已经在 `weight/` 目录中提供训练完成的分类器权重，例如：

```text
weight/
├── mean_major_imbal-40_s1.pt
├── mean_major_imbal-40_s2.pt
├── ...
├── mean_major_imbal-45_s1.pt
├── ...
├── mean_major_imbal-50_s1.pt
└── ...
```

不同文件对应不同温度阈值和随机种子。TemStaPro 会从 `weight/` 目录自动加载对应分类器，因此通过完整 ModelScope 模型包下载后，正常情况下不需要额外下载 TemStaPro 分类器权重。

#### 2）ProtT5-XL-Half-UniRef50

TemStaPro 使用 ProtT5-XL-Half-UniRef50 生成蛋白质序列表征。该模型不包含在当前仓库中，需要单独准备。

```text
Rostlab/prot_t5_xl_half_uniref50-enc
```

推荐将 ProtTrans 模型保存到仓库根目录下的 `ProtTrans/`，运行时通过 `-d/--PT-directory` 指定该目录：

```bash
python scripts/temstapro \
  -f ./scripts/tests/data/long_sequence.fasta \
  -d ./ProtTrans/ \
  --mean-output ./long_sequence_predictions.tsv
```

如果 `./ProtTrans/` 中已经包含以下模型文件，程序会直接从本地加载：

```text
pytorch_model.bin
config.json
tokenizer_config.json
special_tokens_map.json
spiece.model
```

如果指定目录中没有完整模型文件，程序会尝试从 Hugging Face 自动下载并保存到该目录。网络受限或离线环境建议提前下载，可使用 Hugging Face CLI：

```bash
huggingface-cli download \
  Rostlab/prot_t5_xl_half_uniref50-enc \
  --local-dir ./ProtTrans
```

具体网址如下，也可以从页面手动下载所需文件：

```text
https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc/tree/main
```

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/TemStaPro --local_dir ./TemStaPro
cd TemStaPro
```

- TemStaPro 完整推理额外依赖 **ProtT5-XL-Half-UniRef50**；请先按照“权重与模型准备”确认 ProtTrans 模型已经准备完成。
- 仅进行推理时不需要下载 Zenodo 上的训练、验证和测试数据集。

### 快速验证

首先查看命令行参数：

```bash
python scripts/temstapro --help
```

运行仓库保留的官方测试文件：

```bash
make -f scripts/makefile all
```

首次执行测试时可能因为 ProtTrans 模型下载过程导致测试未通过，可清理后重新运行：

```bash
make -f scripts/makefile clean
make -f scripts/makefile all
```

离线环境建议提前准备 ProtTrans 模型后再执行测试。

# 示例数据

官方测试数据位于 `scripts/tests/data/`，主要使用：

```text
scripts/tests/data/long_sequence.fasta
```

作为示例输入。

TemStaPro 输入采用标准 FASTA 格式：

```text
>protein_id
MSEQUENCE...
```

对于自己的预测任务，只需要准备包含一条或多条蛋白质序列的 FASTA 文件，不需要提供蛋白质结构。

# 推理示例

## 蛋白质级热稳定性预测

默认推荐使用 mean embedding 预测：

```bash
python scripts/temstapro \
  -f ./scripts/tests/data/long_sequence.fasta \
  -d ./ProtTrans/ \
  -e ./scripts/tests/outputs/ \
  --mean-output ./long_sequence_predictions.tsv
```

其中：

| 参数 | 说明 |
| --- | --- |
| `-f` | 输入 FASTA 文件 |
| `-d` | ProtTrans/ProtT5 模型目录 |
| `-e` | embedding 缓存目录 |
| `--mean-output` | 蛋白质级预测结果 TSV |

`-e` 不是必须参数，但如果需要多次运行相同序列，建议启用 embedding 缓存。

## 逐残基预测

```bash
python scripts/temstapro \
  -f ./scripts/tests/data/long_sequence.fasta \
  -e ./scripts/tests/outputs/ \
  -d ./ProtTrans/ \
  -p ./ \
  --per-res-output ./long_sequence_predictions_per_res.tsv
```

其中 `-p` 用于指定预测曲线图的输出目录。

## 局部片段预测

TemStaPro 默认使用大小为 41 的窗口进行 per-segment 预测：

```bash
python scripts/temstapro \
  -f ./scripts/tests/data/long_sequence.fasta \
  -e ./scripts/tests/outputs/ \
  -d ./ProtTrans/ \
  --curve-smoothening \
  -p ./ \
  --per-segment-output ./long_sequence_predictions_k41.tsv
```

## 更多温度阈值

如需启用 70、75、80 °C 等附加阈值及 thermophilicity 标签，可以增加：

```bash
--more-thresholds
```

# 输出说明

默认蛋白质级输出为 TSV 表格，包含各温度阈值分类器的 binary prediction 和 raw prediction，并根据多个阈值的结果进一步生成预测温度标签。

默认温度阈值包括：

```text
40
45
50
55
60
65 °C
```

结果中还包含：

```text
clash
```

字段，用于指示多个阈值分类器之间是否出现不一致：

```text
-    无明显冲突
*    存在分类结果不一致
```

如果开启逐残基或局部片段预测，可以进一步生成 TSV 文件；同时指定 `-p` 时会生成 SVG 预测曲线图。

使用 `-e` 时还会在指定目录保存 ProtTrans embedding 缓存文件，可在后续运行中直接复用，从而减少重复的 ProtT5 特征提取开销。

典型运行/中间文件包括：

```text
*.tsv    最终预测结果
*.pt     ProtTrans embedding 缓存
*.svg    逐残基或局部片段预测曲线
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |


# 引用与许可证

- TemStaPro 原始论文：[TemStaPro: protein thermostability prediction using sequence representations from protein language models](https://doi.org/10.1093/bioinformatics/btae157)。
- TemStaPro 官方源码采用 MIT License，详见仓库根目录 `LICENCE.md`。
- TemStaPro 使用 ProtTrans/ProtT5 生成蛋白质表示。使用或再分发对应模型权重时，还需要遵守 ProtTrans、Hugging Face 模型页面及相关预训练数据对应的许可要求。
- 官方训练、验证和测试数据发布于 Zenodo；如使用这些数据开展复现、训练或评测，应按照数据页面要求进行引用。
- 如果在科研工作中使用本仓库，建议同时引用 TemStaPro 原始论文及 OneScience 相关项目信息。
