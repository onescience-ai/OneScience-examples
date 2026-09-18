<p align="center">
  <strong>
    <span style="font-size: 30px;">MVGNN-PPIS</span>
  </strong>
</p>


# 模型介绍

MVGNN-PPIS 是 WW-AILab 开源的蛋白质—蛋白质相互作用位点预测模型，可根据预先生成的蛋白质序列与结构特征，输出每个氨基酸残基属于相互作用位点的概率。模型使用多视图图神经网络联合建模局部序列邻接关系和三维空间邻域关系。

论文：[MVGNN-PPIS: A novel multi-view graph neural network for protein-protein interaction sites prediction based on Alphafold3-predicted structures and transfer learning](https://doi.org/10.1016/j.ijbiomac.2025.140096)

# 模型描述

MVGNN-PPIS 融合 ProtT5 残基表示、DSSP 二级结构特征和 AlphaFold3 预测结构，并通过图卷积与图 Transformer 提取互补的局部和全局信息。模型提供以下主要能力：

- 使用 ProtT5 与 DSSP 组成的 1038 维残基节点特征；
- 根据侧链原子质心坐标构建空间 K 近邻图；
- 使用序列邻接矩阵提取局部残基关系；
- 通过五个官方 checkpoint 进行集成预测；
- 输出逐残基蛋白质相互作用位点概率；
- 在带标签的测试集上计算 AUC、AUPRC、MCC、Accuracy、Precision、Recall 和 F1。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 蛋白质相互作用位点预测 | 根据预计算的序列和结构特征输出逐残基相互作用概率。 |
| Test60 基准评测 | 使用上游提供的 PRO-Test60 数据、预计算特征和五折权重复现评测。 |
| 蛋白质功能位点筛选 | 对候选残基进行概率排序，为后续实验分析提供参考。 |
| DCU 推理验证 | 在 DTK/HIP 版 PyTorch 环境中验证五折模型加载与全量推理。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 需安装pytorch
- 显存占用和运行时间与蛋白质长度、batch size 及并行进程数有关。

### 下载模型包

模型发布到 ModelScope 后，可安装 ModelScope 命令行工具并下载模型包：

```bash
pip install modelscope
modelscope download --model OneScience/MVGNN-PPIS --local_dir ./MVGNN-PPIS
cd MVGNN-PPIS
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

请根据 `requirements.txt` 中声明的依赖版本额外安装依赖：

```bash
conda activate onescience311
python -m pip install -r requirements.txt
```


### 权重与数据准备

基础推理需要五个官方 checkpoint、测试集 CSV 和对应的预计算特征：

| 资产 | 位置 | 用途 |
| --- | --- | --- |
| MVGNN checkpoint | `weight/fold0.ckpt` | 五折集成成员 0 |
| MVGNN checkpoint | `weight/fold1.ckpt` | 五折集成成员 1 |
| MVGNN checkpoint | `weight/fold2.ckpt` | 五折集成成员 2 |
| MVGNN checkpoint | `weight/fold3.ckpt` | 五折集成成员 3 |
| MVGNN checkpoint | `weight/fold4.ckpt` | 五折集成成员 4 |
| Test60 数据 | `weight/datasets/PRO_Test60.csv` | 蛋白质 ID、序列和标签 |
| 预计算特征 | `weight/feature/` | 模型推理输入 |

每个蛋白质 ID 必须对应以下五类 tensor：

```text
weight/feature/<ID>_X.tensor
weight/feature/<ID>_adj.tensor
weight/feature/<ID>_node_feature.tensor
weight/feature/<ID>_mask.tensor
weight/feature/<ID>_label.tensor
```

`scripts/inference.py` 默认根据 `conf/config.json` 加载上述资产，并在推理前检查所有 checkpoint 和特征文件。缺少文件时，脚本会给出缺失数量和文件位置。

基础推理直接使用作者发布的预计算特征，不需要在运行时下载或执行 ProtT5、AlphaFold3 和 DSSP。

### 快速推理

在模型包目录中运行：

```bash
python scripts/inference.py
```

默认配置将使用 `weight/datasets/PRO_Test60.csv` 和 `weight/feature/`，从 `weight/` 加载五个 checkpoint，并将结果写入：

```text
output/prediction/result.csv
output/prediction/test.log
```

查看完整命令行参数：

```bash
python scripts/inference.py --help
```

### 自定义路径与运行参数

推理脚本使用模型包相对路径，不依赖当前工作目录。可修改 `conf/config.json`，也可通过命令行覆盖默认配置：

```bash
python scripts/inference.py \
  --dataset weight/datasets/PRO_Test60.csv \
  --feature-path weight/feature \
  --weight-path weight \
  --output-path output/prediction \
  --device cuda \
  --num-workers 8
```

命令行参数优先于配置文件。自定义数据集仍须提供 `ID`、`sequence` 和 `label` 三列，并为每个 ID 准备五类预计算 tensor。

### 预测结果说明

`result.csv` 保存蛋白质 ID、氨基酸序列和逐残基预测概率。`test.log` 保存模型配置以及带标签数据集的评测指标。



### 特征生成

`scripts/process_feature/` 保留了上游特征生成代码，涉及 ProtT5、AlphaFold3 预测结构和 DSSP。该流程需要额外准备 ProtT5 模型文件、完整结构文件及对应外部工具，不属于基础推理流程。

推荐直接使用作者发布的预计算特征。只有在处理新蛋白质或重新生成特征时，才需要根据上游仓库说明准备 ProtT5、AlphaFold3 和 DSSP 资产。

### 训练

MVGNN-PPIS 上游仓库缺少可直接运行的完整训练入口和完整训练数据，仅提供官方预训练权重及预测代码，因此本模型包不提供训练命令。



# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[MVGNN-PPIS: A novel multi-view graph neural network for protein-protein interaction sites prediction based on Alphafold3-predicted structures and transfer learning](https://doi.org/10.1016/j.ijbiomac.2025.140096)。

- 官方实现：[WW-AILab/MVGNN-PPIS](https://github.com/WW-AILab/MVGNN-PPIS)。截至本模型包整理时，上游仓库未提供明确的 LICENSE 文件；使用、修改或再分发代码、权重和数据前，请向上游作者确认授权范围。

- ProtT5、AlphaFold3、DSSP、数据集及其他第三方资源分别受其原始版权声明、许可证和使用条款约束。

- 模型包顶层的 [`LICENSE.md`](LICENSE.md) 用于记录当前许可状态和第三方资产提示，不构成对上游材料的额外授权。
