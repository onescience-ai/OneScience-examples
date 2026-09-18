<p align="center">
  <strong>
    <span style="font-size: 30px;">ProteinMPNN</span>
  </strong>
</p>

# 模型介绍

ProteinMPNN 是一种基于消息传递神经网络（Message Passing Neural Network）的蛋白质序列设计模型，能够根据给定的蛋白质骨架结构高效生成高表达、可折叠的氨基酸序列。

# 模型描述

ProteinMPNN 采用编码器-解码器架构，编码器通过图神经网络提取骨架结构的几何与拓扑特征，解码器以自回归方式逐位生成氨基酸序列。


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/ProteinMPNN --local_dir ./ProteinMPNN 
cd ProteinMPNN 
```

### 安装运行环境
#### DCU环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

如果运行环境显式需要指向 OneScience 主目录，可设置：

```bash
export ONESCIENCE_ROOT=/path/to/onescience
```

## 快速验证

```bash
export PYTHONPATH=$(pwd)/model:${ONESCIENCE_ROOT}/src:${PYTHONPATH:-}
python -c "from proteinmpnn.protein_mpnn_utils import ProteinMPNN; print('proteinmpnn wrapper ok')"
python scripts/inference.py --help
python scripts/training.py --help
```

## 推理

当前权重已放在 `weight/` 的子目录中。普通 ProteinMPNN 使用 `weight/vanilla_model_weights/`；如果不显式传 `--path_to_model_weights`，`scripts/inference.py` 默认会使用该目录。

### 使用脚本运行最小推理

```bash
cd /path/to/proteinmpnn
bash scripts/test_inference.sh
```

该脚本默认使用：

```text
输入 PDB：data/inputs/PDB_monomers/pdbs/5L33.pdb
设计链：A
模型权重：weight/vanilla_model_weights
输出目录：outputs/test_inference/
```

查看生成序列：

```bash
ls outputs/test_inference/seqs
```

等价命令：

```bash
python scripts/inference.py \
  --pdb_path ./data/inputs/PDB_monomers/pdbs/5L33.pdb \
  --pdb_path_chains "A" \
  --out_folder ./outputs/test_inference \
  --path_to_model_weights ./weight/vanilla_model_weights \
  --model_name v_48_020 \
  --num_seq_per_target 2 \
  --sampling_temp "0.1" \
  --seed 37 \
  --batch_size 1
```

可选权重：

- 普通模型：`--path_to_model_weights ./weight/vanilla_model_weights`
- 可溶蛋白模型：`--path_to_model_weights ./weight/soluble_model_weights` 或加 `--use_soluble_model`
- CA-only 模型：`--path_to_model_weights ./weight/ca_model_weights` 或加`--ca_only`

## 推理示例脚本

`scripts/infer_examples/` 下包含 12 个推理场景示例，均已适配当前目录结构：

| 脚本 | 场景 |
| --- | --- |
| `submit_example_1.sh` | 多个单链 PDB 推理。 |
| `submit_example_2.sh` | 多链复合物，只设计指定链。 |
| `submit_example_3.sh` | 单个 PDB 复合物推理。 |
| `submit_example_3_score_only.sh` | 对已有结构/序列打分，不生成新序列。 |
| `submit_example_3_score_only_from_fasta.sh` | 使用 FASTA 序列对结构打分。 |
| `submit_example_4.sh` | 固定某些残基位置，不设计这些位置。 |
| `submit_example_4_non_fixed.sh` | 只设计指定位置。 |
| `submit_example_5.sh` | tied positions，多位置绑定设计。 |
| `submit_example_6.sh` | 同源寡聚体 homooligomer 约束设计。 |
| `submit_example_7.sh` | 输出 unconditional probabilities。 |
| `submit_example_8.sh` | 加全局氨基酸 bias。 |
| `submit_example_pssm.sh` | 加 PSSM 约束辅助设计。 |

运行单个示例：

```bash
bash scripts/infer_examples/submit_example_3.sh
```

注意：`submit_example_3_score_only_from_fasta.sh` 依赖 `submit_example_3.sh` 先生成 `outputs/example_3_outputs/seqs/3HTN.fa`。

## 训练

当前示例训练数据放在 `data/pdb_2021aug02_sample/`。该目录应包含：

```text
list.csv
valid_clusters.txt
test_clusters.txt
pdb/<pdbid第2-3位>/<pdbid>.pt
pdb/<pdbid第2-3位>/<pdbid>_<chain>.pt
```

直接运行训练示例脚本：

```bash
cd /path/to/proteinmpnn
bash scripts/test_train.sh
```

该脚本默认使用：

```text
训练数据：data/pdb_2021aug02_sample
输出目录：outputs/train/exp_020/
每轮样本数：1000
每 50 轮保存一次 checkpoint
```

查看训练日志和权重：

```bash
cat outputs/train/exp_020/log.txt
ls outputs/train/exp_020/model_weights
```

等价命令：

```bash
python scripts/training.py \
  --path_for_training_data ./data/pdb_2021aug02_sample \
  --path_for_outputs ./outputs/train/exp_020 \
  --num_examples_per_epoch 1000 \
  --save_model_every_n_epochs 50
```

恢复训练时传：

```bash
python scripts/training.py \
  --path_for_training_data ./data/pdb_2021aug02_sample \
  --path_for_outputs ./outputs/train/exp_020 \
  --previous_checkpoint ./outputs/train/exp_020/model_weights/epoch_last.pt
```

## 常用参数

### 推理参数

| 参数 | 说明 | 默认/示例 |
| --- | --- | --- |
| `--pdb_path` | 单个 PDB 输入路径 | `./data/inputs/PDB_monomers/pdbs/5L33.pdb` |
| `--jsonl_path` | 解析后的 PDB JSONL 输入路径 | 由 `parse_multiple_chains.py` 生成 |
| `--pdb_path_chains` | 单 PDB 模式下要设计的链 | `"A"` 或 `"A B"` |
| `--out_folder` | 推理输出目录 | `./outputs/test_inference` |
| `--path_to_model_weights` | 权重目录 | `./weight/vanilla_model_weights` |
| `--model_name` | 权重文件名，不含 `.pt` | `v_48_020` |
| `--num_seq_per_target` | 每个目标生成序列数量 | `2` |
| `--sampling_temp` | 采样温度 | `"0.1"` |
| `--score_only` | 只打分，不生成新序列 | `0` 或 `1` |
| `--save_score` | 保存 score 文件 | `0` 或 `1` |
| `--save_probs` | 保存概率文件 | `0` 或 `1` |
| `--ca_only` | 使用 CA-only 模型 | 默认关闭 |
| `--use_soluble_model` | 使用可溶蛋白模型 | 默认关闭 |

### 训练参数

| 参数 | 说明 | 默认/示例 |
| --- | --- | --- |
| `--path_for_training_data` | 预处理训练数据目录 | `./data/pdb_2021aug02_sample` |
| `--path_for_outputs` | 训练输出目录 | `./outputs/train/exp_020` |
| `--previous_checkpoint` | 恢复训练 checkpoint | `epoch_last.pt` |
| `--num_epochs` | 训练轮数 | 默认 `200` |
| `--num_examples_per_epoch` | 每轮加载样本数 | 示例 `1000` |
| `--batch_size` | token batch size | 默认 `10000` |
| `--save_model_every_n_epochs` | 每隔多少轮保存 checkpoint | 示例脚本为 `50` |
| `--mixed_precision` | 是否使用混合精度 | 默认 `True` |

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 引用与许可证

- ProteinMPNN 原始论文：[Robust deep learning–based protein sequence design using ProteinMPNN](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)。

- ProteinMPNN 相关源码使用 MIT License，详见仓库根目录 `LICENSE`。模型权重和数据的具体使用条款请以对应发布方说明为准。

- 如果在科研工作中使用 ProteinMPNN，建议引用对应ProteinMPNN原始论文和OneScience相关项目信息，并根据实际任务补充下游分析工具或数据集引用。

