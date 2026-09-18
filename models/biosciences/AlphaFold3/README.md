<p align="center">
  <strong>
    <span style="font-size: 30px;">AlphaFold3</span>
  </strong>
</p>

# 模型介绍

AlphaFold3 是 Google DeepMind 和 Isomorphic Labs 提出的生物分子结构预测模型，可预测蛋白质、DNA、RNA、小分子配体等分子及其复合物的三维结构与相互作用。

论文：Accurate structure prediction of biomolecular interactions with AlphaFold 3  
https://www.nature.com/articles/s41586-024-07487-w

# 模型描述

AlphaFold3 采用 Pairformer 与扩散模型预测生物分子复合物结构。本模型包提供 JAX / Flax 推理工程和数据搜索脚本，并配套发布 ModelScope 数据集 `OneScience/AlphaFold3_dataset`。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 已有特征直接推理 | 输入包含 MSA / template 等特征的 AlphaFold3 JSON，输出结构预测结果 |
| 蛋白结构预测 | 输入蛋白序列，结合搜索数据库生成特征并预测结构 |
| 生物分子复合物建模 | 输入蛋白、DNA、RNA、配体等多组分对象，预测复合物空间构象 |
| 数据搜索流程验证 | 使用 Jackhmmer / Nhmmer 或 MMseqs 流程检查数据库路径和搜索工具连通性 |
| ModelScope / OneCode 运行 | 下载模型工程和完整数据集后，在生物领域运行环境中快速验证脚本连通性 |



# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。





**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光 DCU：

```bash
hy-smi
```

### 下载模型包

```bash
modelscope download --model OneScience/AlphaFold3 --local_dir ./AlphaFold3
cd AlphaFold3
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

安装完成后回到模型包目录：

```bash
cd ./AlphaFold3
```

如当前环境尚未构建 AlphaFold3 C++ 扩展和运行数据文件，可执行：

```bash
python -m onescience.flax_model.alphafold3.build_extension
python -m onescience.flax_models.alphafold3.build_data
```

### 训练与推理数据介绍

OneScience 社区已将 AlphaFold3 推理与数据搜索所需的完整数据上传至 ModelScope：[OneScience/AlphaFold3_dataset](https://modelscope.cn/datasets/OneScience/AlphaFold3_dataset)。本模型包不包含训练入口，该数据集主要用于 MSA / template 特征构建和推理前的数据搜索。

```bash
modelscope download --dataset OneScience/AlphaFold3_dataset --local_dir ./data/alphafold3
```
### 训练权重

权重即将上传

### 准备权重

请将 AlphaFold3 模型权重放置在以下目录，或通过环境变量指定：

```text
weight/
  AlphaFold3/
    ...
```

默认查找顺序为：

- `ALPHAFOLD3_MODEL_DIR`
- `${ONESCIENCE_MODELS_DIR}/AlphaFold3`
- `weight/AlphaFold3`

示例：

```bash
export ALPHAFOLD3_MODEL_DIR=/path/to/AlphaFold3
```

### 直接推理

当输入 JSON 已包含 MSA、template 等特征时，可直接运行：

```bash
bash scripts/infer.sh
```

等价的 Python 命令示例：

```bash
python scripts/run_alphafold.py \
  --json_path inputs/7r6r_data.json \
  --model_dir weight/AlphaFold3 \
  --output_dir outputs \
  --run_data_pipeline=false \
  --flash_attention_implementation=triton
```

输出会写入 `outputs/`，包含最佳结构、不同 seed / sample 的结构结果、ranking score CSV 和输入 JSON 副本。

### Jackhmmer / Nhmmer 数据搜索

当输入 JSON 仅包含序列、需要本地数据库搜索时，可使用：

```bash
bash scripts/infer_jackhmmer.sh
```

常用环境变量：

```bash
export ALPHAFOLD3_DATASET_ROOT=/path/to/alphafold3
export ALPHAFOLD3_MODEL_DIR=/path/to/AlphaFold3
export ALPHAFOLD3_JSON_PATH=inputs/t1119_search.json
export ALPHAFOLD3_OUTPUT_DIR=outputs
export ALPHAFOLD3_RUN_INFERENCE=false
```

其中 `ALPHAFOLD3_DATASET_ROOT` 默认需要包含 `public_databases/`、`jackhmmer_split/` 和 `mmseqsDB/` 等数据库目录。

### MMseqs 数据搜索

如运行环境提供 MMseqs 程序和 MMseqs 数据库，可使用：

```bash
bash scripts/infer_mmseqs.sh
```

常用环境变量：

```bash
export ALPHAFOLD3_MMSEQS_HOME=/path/to/mmseqs
export ALPHAFOLD3_DATASET_ROOT=/path/to/alphafold3
export ALPHAFOLD3_MMSEQS_DB_DIR=/path/to/alphafold3/mmseqsDB
export ALPHAFOLD3_RUN_INFERENCE=false
```

如需搜索后继续推理，可将 `ALPHAFOLD3_RUN_INFERENCE` 设为 `true`，并确保权重目录可用。

# 数据格式

AlphaFold3 输入采用 JSON 格式，基本结构如下：

```json
{
  "dialect": "alphafold3",
  "version": 1,
  "name": "example",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "..."
      }
    }
  ],
  "modelSeeds": [100],
  "bondedAtomPairs": null,
  "userCCD": null
}
```

本仓库提供两个示例：

- `inputs/7r6r_data.json`：包含序列、MSA 和 template 等信息，适合直接推理。
- `inputs/t1119_search.json`：仅包含序列，适合数据搜索流程验证。

ModelScope 完整数据集 `OneScience/AlphaFold3_dataset` 建议下载到模型包下的 `data/alphafold3/`。数据搜索流程默认读取的相对结构如下：

```text
data/
  alphafold3/
    public_databases/
      mmcif_files/
      pdb_seqres_2022_09_28.fasta
      ...
    jackhmmer_split/
      bfd-first_non_consensus_sequences.fasta@64
      mgy_clusters_2022_05.fa@512
      uniprot_cluster_annot_2021_04.fa@256
      uniref90_2022_05.fa@128
    mmseqsDB/
      small_bfd_db
      mgnify_db
      uniprot_cluster_annot_db
      uniref90_db
```

# 验证

静态导入检查：

```bash
python tests/check_import_boundaries.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库基于 AlphaFold3 开源模型进行 DCU 适配。
- AlphaFold3 源码使用 CC BY-NC-SA 4.0 许可；模型参数受独立使用条款约束。
- 科研使用请引用原始论文：[Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)。
