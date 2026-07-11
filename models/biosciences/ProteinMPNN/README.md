<h1 align="center">ProteinMPNN</h1>

## 模型介绍

ProteinMPNN 是用于蛋白质序列设计的图神经网络模型，可根据蛋白质骨架结构设计氨基酸序列。本目录是 OneScience 中 ProteinMPNN 的轻量调用包，包含模型源码、推理脚本、训练脚本、辅助脚本、示例输入、示例训练数据和权重目录。

## 仓库说明

本仓库不是完整 standalone 版本。`model/proteinmpnn/` 包含 ProteinMPNN 模型相关源码，`scripts/` 包含推理、训练和辅助处理脚本；其余环境和公共依赖继续由 OneScience 基座提供。

当前支持：

- 根据单链或多链 PDB 骨架生成候选氨基酸序列。
- 对结构/序列组合进行 score-only 打分。
- 输出 conditional probability 或 unconditional probability。
- 使用固定链、固定残基、tied positions、PSSM、氨基酸 bias 等约束进行设计。
- 使用 `data/pdb_2021aug02_sample/` 中的 ProteinMPNN 预处理样例数据验证训练流程。

当前不直接支持：

- 结构预测、分子动力学模拟或湿实验验证。
- 直接用普通 PDB 文件训练；训练入口需要 ProteinMPNN 预处理后的 `.pt` 数据集结构。


## 文件说明

| 路径 | 类型 | 作用 | 备注 |
|---|---|---|---|
| `README.md` | 文档 | 人类用户和大模型入口 | 中文使用说明 |
| `configuration.json` | 元信息 | 声明模型名、任务、入口、源码包、配置和权重目录 | ModelScope/OneScience 资源描述 |
| `config/config.yaml` | 配置 | 记录模型包路径、推理/训练入口、默认数据和权重目录 | 可供自动化系统读取 |
| `model/proteinmpnn/` | Python 包 | ProteinMPNN 模型、特征、数据加载和训练工具 | 本地源码 |
| `scripts/inference.py` | 推理脚本 | 序列设计、打分和概率输出入口 | 支持 PDB 或 JSONL 输入 |
| `scripts/training.py` | 训练脚本 | ProteinMPNN 训练入口 | 需要预处理训练数据 |
| `scripts/test_inference.sh` | 验证脚本 | 单 PDB 最小推理示例 | 依赖 `data/inputs` 和 `weight/vanilla_model_weights` |
| `scripts/test_train.sh` | 验证脚本 | 最小训练示例 | 依赖 `data/pdb_2021aug02_sample` |
| `scripts/helper_scripts/` | 辅助脚本 | 解析 PDB、指定设计链、固定位置、PSSM、bias、tied positions 等 | 供复杂推理示例调用 |
| `scripts/infer_examples/` | 示例脚本 | 12 个推理场景示例 | 
| `data/` | 数据目录 | 放置推理示例输入和训练样例数据 | 当前目录为空，需下载后运行 |
| `weight/` | 权重目录 | 放置 vanilla、soluble、CA-only 模型权重 | 当前目录为空，需下载后运行 |
| `outputs/` | 输出目录 | 推理、训练日志和 checkpoint 输出 | 运行时自动生成 |

## 环境安装

推荐在 GPU 或 DCU 环境运行。CPU 可用于导入检查、小规模连通性验证和查看脚本帮助，完整推理和训练速度较慢。DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

安装 OneScience 生物方向环境：

```bash
git clone https://gitee.com/onescience-ai/onescience.git
cd onescience
bash install.sh bio
```

进入 ProteinMPNN 运行包目录后，如果 OneScience 主仓库不在默认相对位置，可显式设置：

```bash
export ONESCIENCE_ROOT=/path/to/onescience
export PYTHONPATH=$(pwd)/model:${ONESCIENCE_ROOT}/src:${PYTHONPATH:-}
```

如果只需要使用本目录中的 ProteinMPNN 源码，也可以设置：

```bash
export PYTHONPATH=$(pwd)/model:${PYTHONPATH:-}
```

## 文件下载

权重下载：

```bash
modelscope download \
  --model OneScience/ProteinMPNN \
  --include 'weight/**' \
  --local_dir /path/to/ProteinMPNN
```

数据下载：

```bash
modelscope download \
  --model OneScience/ProteinMPNN \
  --include 'data/**' \
  --local_dir /path/to/ProteinMPNN
```

下载后建议确认以下目录存在：

```text
data/inputs/
data/pdb_2021aug02_sample/
weight/vanilla_model_weights/
weight/soluble_model_weights/
weight/ca_model_weights/
```

普通模型权重文件通常包括 `v_48_002.pt`、`v_48_010.pt`、`v_48_020.pt`、`v_48_030.pt`。`scripts/inference.py` 默认使用 `v_48_020`。

## 快速验证

先检查 Python 包和脚本是否可导入：

```bash
python -c "from proteinmpnn.protein_mpnn_utils import ProteinMPNN; print('proteinmpnn import ok')"
python scripts/inference.py --help
python scripts/training.py --help
```

## 推理

普通 ProteinMPNN 使用 `weight/vanilla_model_weights/`；如果不显式传 `--path_to_model_weights`，`scripts/inference.py` 默认会使用该目录。

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
- CA-only 模型：`--path_to_model_weights ./weight/ca_model_weights --ca_only`

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

ProteinMPNN 原始方法来自蛋白质序列设计相关论文和开源实现。本目录按 OneScience 标准运行包整理代码、权重、示例和运行元数据；具体许可证以仓库内原始文件和 ModelScope 页面声明为准。
