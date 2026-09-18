<p align="center">
  <strong><span style="font-size: 30px;">SurfDock</span></strong>
</p>

# 模型介绍

SurfDock 是一种面向蛋白质-配体复合物预测与结构基础虚拟筛选的表面信息增强扩散生成模型。模型将蛋白质表面几何与化学信息引入扩散式对接流程，用于生成并筛选蛋白质-小分子结合构象。

论文：

> **SurfDock is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction**  
> Duanhua Cao, Mingan Chen, Rui Zhang, et al.  
> *Nature Methods*, 2024  
> DOI: https://doi.org/10.1038/s41592-024-02516-y

# 模型描述

SurfDock 是一种面向蛋白质-配体复合物预测与结构基础虚拟筛选的表面信息增强扩散生成模型。模型首先对目标蛋白质结构进行预处理，并计算蛋白质表面的几何与理化信息；同时利用 ESM 提取蛋白质序列表征，从而为后续配体构象生成提供结构与序列两方面的信息。

在推理阶段，SurfDock 通过扩散生成模型对配体在蛋白质结合位点中的候选构象进行采样，并结合姿态置信度模型对生成结果进行评估与排序。对于虚拟筛选任务，还可以进一步使用筛选评分模型对候选蛋白质-配体构象进行重打分，从而获得更适合后续排序和筛选的结果。

通过引入蛋白质表面信息、蛋白质语言模型表征以及扩散式构象生成机制，SurfDock 可用于蛋白质-配体对接、候选结合姿态生成、姿态评分以及结构基础虚拟筛选等任务。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 蛋白质-配体对接 | 预测配体在蛋白质结合位点中的结合构象 |
| 结构基础虚拟筛选 | 对小分子库进行批量 docking 与评分 |
| 配体构象生成 | 基于扩散模型采样候选结合姿态 |
| 蛋白质表面建模 | 使用蛋白质表面几何与理化信息辅助 docking |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

### 硬件要求

- SurfDock 的扩散采样、ESM 表征提取以及图神经网络计算具有较高计算量，推荐使用 GPU/DCU 加速设备。

### 环境安装

#### DCU 环境

```bash
# 请首先激活 DTK 及 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

pip install onescience[bio] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

### 环境说明

- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考仓库根目录下的 `environment.yaml` 声明的依赖版本，补充安装或调整相应依赖。
- SurfDock 的蛋白质表面处理流程依赖 PyMesh。若当前 Python 版本与上游 PyMesh 不完全匹配，应根据实际调用关系进行兼容适配。


### 权重与数据准备

#### SurfDock 模型权重

当前仓库已经包含推理所需的主要模型权重：

```text
weight/
├── docking/
│   ├── best_ema_inference_epoch_model.pt
│   └── model_parameters.yml
├── posepredict/
│   ├── best_model.pt
│   └── model_parameters.yml
└── screen/
    ├── best_model.pt
    └── model_parameters.yml
```

因此完整下载仓库后，一般不需要再次单独下载 SurfDock 主模型权重。

检查：

```bash
ls -lh weight/docking/
ls -lh weight/posepredict/
ls -lh weight/screen/
```

#### ESM 模型

SurfDock 使用 ESM 提取蛋白质序列表征。

官方安装方式：

```bash
git clone https://github.com/facebookresearch/esm model/esm
cd model/esm
pip install -e .
cd ../..
```

官方推理脚本使用：

```text
esm2_t33_650M_UR50D
```

并通过：

```bash
python model/esm/scripts/extract.py \
  "esm2_t33_650M_UR50D" \
  input.fasta \
  output_dir \
  --repr_layers 33 \
  --include "per_tok" \
  --truncation_seq_length 4096
```

提取蛋白质 residue-level embedding。

- 若当前环境无法联网，建议提前缓存对应 ESM 权重。`esm2_t33_650M_UR50D` 需要准备以下两个文件：

```text
https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

下载后建议放置到当前用户的 Torch Hub checkpoints 缓存目录：

```text
~/.cache/torch/hub/checkpoints/
```

最终文件路径应为：

```text
~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt
```

#### PDBBind 数据

如果需要重新训练 SurfDock，则必须额外获取 PDBBind。官方说明此前由 EquiBind 提供的预处理数据因 PDBBind 许可证限制不再公开分发，因此需要用户自行从 PDBBind 官方渠道获取并处理。

处理完成后放置在：

```text
model/data/PDBBind_processed/
```
`model/data/splits/` 保存的是数据划分信息，不等同于完整 PDBBind 数据。

## 3. 快速开始

### 下载模型包

```bash
modelscope download \
  --model OneScience/SurfDock \
  --local_dir ./SurfDock

cd SurfDock
```
- SurfDock 使用 ESM 提取蛋白质序列表征，因此需要额外下载 ESM 模型。详见模型与数据准备小节内容。

# 示例数据

当前仓库已经提供 docking 与 screening 示例：

```text
model/data/eval_sample_dirs/
model/data/Screen_sample_dirs/
```

用户使用自己的数据时，需要按照示例目录组织蛋白质与配体输入，并修改相应 bash 脚本中的：

```text
data_dir
surface_out_dir
out_csv_file
Screen_lib_path
docking_out_dir
```

等路径。

# 推理示例

以下命令默认在 SurfDock 仓库根目录执行。

## 蛋白质-配体对接示例

执行：

```bash
cd scripts/bash_scripts/test_scripts
bash eval_samples.sh
```

脚本主要自动完成：

```text
1. 蛋白质结构预处理
2. 计算蛋白质表面
3. 构建推理输入 CSV
4. 提取 ESM embedding
5. 运行 SurfDock 扩散采样
6. 保存 docking 结果
```

运行前建议检查：

```bash
vim scripts/bash_scripts/test_scripts/eval_samples.sh
```

重点确认：

```text
gpu_string
data_dir
surface_out_dir
out_csv_file
esmbedding_dir
docking_out_dir
```

用户还需要根据实际适配方式调整脚本中的 `CUDA_VISIBLE_DEVICES` 与 `accelerate launch` 相关设置。

## 虚拟筛选示例

执行：

```bash
cd scripts/bash_scripts/test_scripts
bash screen_pipeline.sh
```

运行前检查：

```bash
vim scripts/bash_scripts/test_scripts/screen_pipeline.sh
```

重点修改：

```text
gpu_string
data_dir
surface_out_dir
out_csv_file
esmbedding_dir
Screen_lib_path
docking_out_dir
```

其中 `Screen_lib_path` 指定待筛选的小分子库，例如官方示例中的：

```text
model/data/Screen_sample_dirs/test_samples/1a0q/1a0q_ligand_for_Screen.sdf
```

筛选流程主要为：

```text
蛋白质预处理
        ↓
蛋白质表面计算
        ↓
ESM embedding
        ↓
SurfDock 生成候选构象
        ↓
screen 模型重新评分
        ↓
输出筛选结果
```

## 跳过已完成的蛋白质预处理

在以下两个脚本中可以修改该参数：

```text
scripts/bash_scripts/test_scripts/eval_samples.sh
scripts/bash_scripts/test_scripts/screen_pipeline.sh
```

脚本通过：

```bash
target_have_processed=true
```

控制是否跳过 target preprocessing。设置为 `true` 时，脚本会跳过 OpenBabel/reduce 等目标蛋白预处理步骤，直接进入后续表面计算、CSV 构建、ESM embedding 和推理流程。

若需要重新处理目标蛋白：

```bash
target_have_processed=false
```

设置为 `false` 时，脚本会重新执行目标蛋白预处理步骤。

## ESM embedding 单独生成

首先构建 FASTA：

```bash
python model/datasets/esm_embedding_preparation.py \
  --out_file ./protein.fasta \
  --protein_ligand_csv ./input.csv
```

提取 ESM 表征：

```bash
python model/esm/scripts/extract.py \
  "esm2_t33_650M_UR50D" \
  ./protein.fasta \
  ./esm_embedding_output \
  --repr_layers 33 \
  --include "per_tok" \
  --truncation_seq_length 4096
```

提取 pocket embedding：

```bash
python model/datasets/get_pocket_embedding.py \
  --protein_pocket_csv ./input.csv \
  --embeddings_dir ./esm_embedding_output \
  --pocket_emb_save_dir ./esm_embedding_pocket_output
```

合并成 SurfDock 推理需要的 `.pt`：

```bash
python model/datasets/esm_pocket_embeddings_to_pt.py \
  --esm_embeddings_path ./esm_embedding_pocket_output \
  --output_path ./esm2_pocket_embeddings.pt
```

# 训练说明

## SurfDock 重新训练

重新训练需要先准备 PDBBind 数据，并完成蛋白质表面和 ESM embedding 预处理。

训练相关脚本位于：

```text
scripts/bash_scripts/train_SurfDock_docking_module/
```

先按照“ESM embedding 单独生成”小节准备训练需要的 ESM embedding 文件。当前仓库的训练脚本位于：

```text
scripts/bash_scripts/train_SurfDock_docking_module/train_SurfDock.sh
```

完成 ESM embedding 后，检查 `train_SurfDock.sh` 中的数据、模型和输出路径，然后执行：

```bash
cd scripts/bash_scripts/train_SurfDock_docking_module
bash train_SurfDock.sh
```

## SurfScore 重新训练

SurfScore 训练相关脚本位于：

```text
scripts/bash_scripts/train_SurfScore/train_SurfScore.sh
```

运行前需要检查脚本中的 PDBBind 数据、cache、ESM embedding、输出目录和 GPU 参数，然后执行：

```bash
cd scripts/bash_scripts/train_SurfScore
bash train_SurfScore.sh
```

# 输出说明

## Docking 输出

`eval_samples.sh` 的 docking 结果保存在脚本中 `docking_out_dir` 指定的目录。

主要输出包括：

```text
生成的蛋白质-配体构象
置信度/评分结果
运行日志
中间 CSV
ESM embedding
蛋白质表面文件
```

## Screening 输出

`screen_pipeline.sh` 会在 `docking_out_dir` 中生成筛选结果，并使用：

```text
weight/screen/best_model.pt
```

对 docking pose 重新评分。

最终结果可用于候选小分子的排序和筛选。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |


# 引用与许可证

- SurfDock 官方源码仓库采用 **MIT License**，允许使用、修改、分发、再许可和商业使用；复制或分发时应保留原始版权声明和 MIT License 许可证文本。
- PDBBind 数据受其自身许可证和使用条款约束，SurfDock 的 MIT License 不自动覆盖 PDBBind 数据。
- 本仓库为 SurfDock 的 **DCU 适配版本**，对部分运行环境、依赖配置和执行方式进行了调整；仓库代码、模型权重及相关数据的使用仍应以各自原始项目中的许可证及使用条款为准。
