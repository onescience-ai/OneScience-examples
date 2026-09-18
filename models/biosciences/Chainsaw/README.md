<p align="center">
  <strong>
    <span style="font-size: 30px;">Chainsaw</span>
  </strong>
</p>

# 模型介绍

Chainsaw 是一个根据蛋白质三维结构预测结构域边界的全卷积神经网络。模型从 PDB 或 mmCIF 结构中提取残基距离和 STRIDE 二级结构特征，预测残基对属于同一结构域的概率，并通过后处理生成结构域切分结果。

论文：[Chainsaw: protein domain segmentation with fully convolutional neural networks](https://doi.org/10.1093/bioinformatics/btae296)

# 模型描述

Chainsaw 的输入是蛋白质三维结构，而不是单独的氨基酸序列。主要处理流程如下：

1. 解析指定链并生成残基距离矩阵；
2. 调用 STRIDE 计算二级结构特征；
3. 使用全卷积神经网络预测残基对的结构域共归属关系；
4. 将预测矩阵转换为连续或不连续的结构域边界，并输出置信度。

模型包包含三版官方预训练权重，默认使用 `model_v3`。基础推理不需要联网下载额外权重，也不需要重新训练模型。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 单结构预测 | 对一个 PDB 或 mmCIF 文件中的蛋白质链进行结构域切分。 |
| 批量结构预测 | 批量处理目录中的 PDB 和 mmCIF 结构文件。 |
| AlphaFold 结构分析 | 对 AlphaFold 预测结构进行结构域边界识别。 |
| 结构域边界筛选 | 输出结构域数量、残基区间、置信度和运行时间，供后续分析使用。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 支持 CPU 和 DCU 推理；

- STRIDE、结构解析和部分后处理在 CPU 上执行；


### 下载模型包

安装 ModelScope 命令行工具后下载模型：

```bash
pip install modelscope
modelscope download --model OneScience/Chainsaw --local_dir ./Chainsaw
cd Chainsaw
```

### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```



### 编译 STRIDE

Chainsaw 依赖 STRIDE 生成二级结构特征。STRIDE 源码位于 `scripts/stride/`，首次使用前执行：

```bash
cd scripts/stride
make
chmod +x stride
cd ../..
```

推理入口默认查找 `scripts/stride/stride`。从模型包根目录运行命令时，无需额外设置路径；如需显式指定，可使用相对路径：

```bash
export STRIDE_EXE=scripts/stride/stride
```

### 权重与数据准备

模型包已包含基础推理所需的官方权重和配置，无需在运行时额外下载：

| 模型版本 | 权重 | 配置 |
| --- | --- | --- |
| model_v1 | `weight/model_v1/weights.pt` | `conf/model_v1/` |
| model_v2 | `weight/model_v2/weights.pt` | `conf/model_v2/` |
| model_v3（默认） | `weight/model_v3/weights.pt` | `conf/model_v3/` |

输入数据由用户提供，支持 PDB 和 mmCIF 结构文件。模型包中的 `scripts/example_files/` 可用于快速验证，无需下载训练数据集。

### 快速推理

以下命令均应在模型包根目录运行。输出目录不存在时，推理脚本会自动创建。

**GPU 推理**

```bash
unset CUDA_VISIBLE_DEVICES
export TORCHDYNAMO_DISABLE=1

python scripts/get_predictions.py \
  --structure_file scripts/example_files/AF-A0A1W2PQ64-F1-model_v4.pdb \
  --output output/inference/predictions_dcu.tsv
```

**CPU 推理**

```bash
export CUDA_VISIBLE_DEVICES=""
export TORCHDYNAMO_DISABLE=1

python scripts/get_predictions.py \
  --structure_file scripts/example_files/AF-A0A1W2PQ64-F1-model_v4.pdb \
  --output output/inference/predictions_cpu.tsv
```

**批量处理结构目录**

将待预测文件放入 `input/structures/`，然后执行：

```bash
python scripts/get_predictions.py \
  --structure_directory input/structures \
  --output output/inference/batch_predictions.tsv
```

**使用其他官方模型版本**

```bash
python scripts/get_predictions.py \
  --model_dir weight/model_v1 \
  --config_dir conf/model_v1 \
  --structure_file input/protein.pdb \
  --output output/inference/model_v1_predictions.tsv
```

### 输出说明

推理结果为 TSV 文件，每个结构链对应一行：

| 字段 | 含义 |
| --- | --- |
| `chain_id` | 输入结构链标识；未显式指定链时使用首条链。 |
| `sequence_md5` | 输入氨基酸序列的 MD5。 |
| `nres` | 有效残基数。 |
| `ndom` | 预测结构域数量。 |
| `chopping` | 结构域残基区间；下划线连接同一不连续结构域的片段，逗号分隔不同结构域。 |
| `confidence` | 预测置信度。 |
| `time_sec` | 单个结构的推理耗时，单位为秒。 |




### 训练

因上游仓库未公开完整训练数据、可执行训练入口，以及学习率、批大小、epoch 等完整训练参数，本模型包不提供训练命令。官方预训练权重可直接完成目标推理。


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[Chainsaw: protein domain segmentation with fully convolutional neural networks](https://doi.org/10.1093/bioinformatics/btae296)。
- 官方实现：[JudeWells/chainsaw](https://github.com/JudeWells/chainsaw)，Chainsaw 代码采用 MIT License。

- STRIDE 及其他第三方组件分别受其原始版权声明、许可证和使用条款约束。
