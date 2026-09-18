<p align="center">
  <strong>
    <span style="font-size: 30px;">EpHod</span>
  </strong>
</p>

# 模型介绍

EpHod 是一个用于预测酶催化最适 pH（pHopt）的集成模型。它首先使用 ESM-1v 将氨基酸序列编码为蛋白质表示，再综合残差轻量注意力网络 RLATtr 与支持向量回归模型 SVR 的预测结果。

论文：[Machine learning prediction of enzyme optimum pH](https://doi.org/10.1038/s42256-025-01026-6)



# 模型描述

EpHod 推理链路包含三个部分：

- ESM-1v：将酶序列编码为逐残基 1280 维表示；
- RLATtr：通过残差轻量注意力网络回归 pHopt，同时可输出注意力权重和 2560 维 EpHod 表示；
- SVR：对池化、标准化后的 ESM-1v 表示进行支持向量回归；
- Ensemble：取 RLATtr 与 SVR 预测的平均值作为最终结果。

官方 RLATtr 先在约 190 万条带最适环境 pH（pHenv）标签的蛋白质上预训练，再在 9855 条带催化最适 pH（pHopt）标签的酶数据上微调。输入序列超过 1022 个残基时会截断；为避免池化偏差，当前推理入口固定使用 batch size 1。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 酶最适 pH 预测 | 根据酶的氨基酸序列预测催化最适 pH。 |
| 酶候选初筛 | 对多条候选序列进行相对比较。 |
| 注意力分析 | 可选保存 RLATtr 的逐残基注意力权重。 |
| 蛋白质表征提取 | 可选保存 RLATtr 最终层的 2560 维表示。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 支持 CPU 和 PyTorch 支持的加速设备；
- 推荐使用 GPU 或 SCNet DCU 运行 ESM-1v 推理，CPU 可以运行但速度较慢；
- ESM-1v 为 650M 参数模型，实际显存占用随序列长度变化，显存不足时应缩短输入序列或逐条推理。

### 下载模型包

安装 ModelScope 命令行工具后下载模型：

```bash
python -m pip install modelscope
modelscope download --model OneScience/EpHod --local_dir ./EpHod
cd EpHod
```



### 安装运行环境

**SCNet DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
python -m pip install "onescience[bio-dcu]" \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

在 OneScience 环境基础上安装 EpHod 额外依赖：

```bash
python -m pip install --no-deps -r requirements.txt
```


### 权重准备

推理需要以下三个文件，缺一不可：

| 资产 | 相对位置 | 用途 |
| --- | --- | --- |
| ESM-1v 650M 权重 | `weight/esm1v_t33_650M_UR90S_1.pt` | 生成逐残基蛋白质表示。 |
| RLATtr 权重 | `weight/ESM1v-RLATtr.pt` | 神经网络分支预测。 |
| SVR 模型及标准化统计量 | `weight/ESM1v-SVR.pkl` | 传统机器学习分支预测。 |

官方来源：

- [ESM-1v 主权重](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt)
- [EpHod RLATtr 权重与训练数据](https://doi.org/10.5281/zenodo.14252615)
- `ESM1v-SVR.pkl` 随 EpHod 官方仓库发布。



### 快速推理

以下命令使用一条已验证的 smoke 序列：

```bash
python scripts/inference.py \
  --fasta_path conf/data/smoke.fasta \
  --output_path output/smoke/prediction.csv \
  --verbose 1 \
  --save_attention_weights 0 \
  --save_embeddings 0
```

完整官方示例：

```bash
python scripts/inference.py \
  --fasta_path conf/data/test_sequences.fasta \
  --output_path output/inference/prediction.csv \
  --verbose 1 \
  --save_attention_weights 0 \
  --save_embeddings 0
```

结果 CSV 包含三个预测列：

```text
RLATtr,SVR,Ensemble
```

`--output_path` 用于直接指定预测 CSV 的完整路径，并会自动创建父目录。原有的 `--save_dir` 与 `--csv_name` 参数继续保留；未提供 `--output_path` 时，输出路径仍由这两个参数组合生成。



### 保存注意力权重和模型表示

将对应开关设为 1：

```bash
python scripts/inference.py \
  --fasta_path conf/data/smoke.fasta \
  --output_path output/features/prediction.csv \
  --save_attention_weights 1 \
  --save_embeddings 1
```

输出包括 `attention_weights/`、`embeddings.csv` 和预测 CSV。



# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- EpHod 论文：[Machine learning prediction of enzyme optimum pH](https://doi.org/10.1038/s42256-025-01026-6)
- 官方实现：[jafetgado/EpHod](https://github.com/jafetgado/EpHod)，采用 [MIT License](https://github.com/jafetgado/EpHod/blob/main/LICENSE)。
- 模型与数据：[Machine learning prediction of enzyme optimal pH](https://doi.org/10.5281/zenodo.14252615)。
- 本模型包是在官方实现基础上完成的 SCNet 运行适配与目录整理，不改变论文、官方代码、模型权重、数据集及第三方资源各自的版权声明、许可证和使用条款。
