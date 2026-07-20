<p align="center">
  <strong>
    <span style="font-size: 30px;">AlphaFold3</span>
  </strong>
</p>

# 模型介绍

AlphaFold3 是 Google DeepMind 和 Isomorphic Labs 提出的生物分子结构预测模型，可预测蛋白质、DNA、RNA、小分子配体、离子和翻译后修饰等多类型分子之间的三维结构与相互作用。相比仅面向蛋白单体或蛋白复合物的结构预测流程，AlphaFold3 进一步覆盖蛋白-核酸、蛋白-配体、核酸复合物等更广泛的生物分子体系，适用于复合物建模、候选分子机制分析、结构生物学验证前处理和下游分子设计场景。

本模型包提供 AlphaFold3 的 JAX / Flax 推理工程、输入 JSON 示例、数据搜索流程脚本和本地推理启动脚本，可作为独立工程下载、部署和运行。

论文：Accurate structure prediction of biomolecular interactions with AlphaFold 3  
https://www.nature.com/articles/s41586-024-07487-w

# 仓库说明

本仓库是 AlphaFold3 最小可运行独立模型仓库，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用已包含 MSA / template 等特征的 AlphaFold3 JSON 输入进行结构推理
- 使用 Jackhmmer / Nhmmer 数据搜索流程生成输入特征
- 使用 MMseqs 数据搜索流程生成输入特征
- 支持蛋白、核酸、配体等 AlphaFold3 JSON 输入对象
- 输出结构文件、ranking score、置信度和完整推理结果
- 支持通过环境变量指定权重、数据库、输入和输出目录

当前不支持能力：

- 不提供结构可视化服务或实验结果自动判读
- 不面向临床诊断或医学决策

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 已有特征直接推理 | 输入包含 MSA / template 等特征的 AlphaFold3 JSON，输出结构预测结果 |
| 蛋白结构预测 | 输入蛋白序列，结合搜索数据库生成特征并预测结构 |
| 生物分子复合物建模 | 输入蛋白、DNA、RNA、配体等多组分对象，预测复合物空间构象 |
| 数据搜索流程验证 | 使用 Jackhmmer / Nhmmer 或 MMseqs 流程检查数据库路径和搜索工具连通性 |
| OneCode / 本地运行 | 在生物领域运行环境中快速验证脚本连通性 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `flax_model/alphafold3/` | AlphaFold3 模型源码 | 包含模型、数据管线、结构处理和 C++ 扩展构建入口 |
| `flax_model/alphafold3/model/` | 模型网络、特征、置信度和后处理模块 | 推理核心实现 |
| `flax_model/alphafold3/data/` | MSA、template、数据库搜索和特征构建模块 | 供数据搜索流程调用 |
| `flax_model/alphafold3/structure/` | mmCIF、化学组分、键合与结构表处理模块 | 用于结构读写和后处理 |
| `scripts/run_alphafold.py` | 主运行脚本 | 支持数据管线和模型推理 |
| `scripts/infer.sh` | 直接推理启动脚本 | 默认读取 `inputs/7r6r_data.json` |
| `scripts/infer_jackhmmer.sh` | Jackhmmer / Nhmmer 搜索流程启动脚本 | 需要公共数据库 |
| `scripts/infer_mmseqs.sh` | MMseqs 搜索流程启动脚本 | 需要 MMseqs 程序和数据库 |
| `inputs/7r6r_data.json` | 已含特征的示例输入 | 用于直接推理 |
| `inputs/t1119_search.json` | 仅序列示例输入 | 用于数据搜索流程 |
| `weight/` | 权重占位目录 | 默认查找 `weight/AlphaFold3` |
| `tests/check_import_boundaries.py` | 静态导入检查脚本 | 用于工程完整性验证 |
| `LICENSE` | 许可证说明 | 源码和模型参数遵循各自使用条款 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**软件要求**

请参考 OneScience 生物领域运行环境，DCU 用户想了解更多适配内容请联系 liubiao@sugon.com。

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光 DCU：

```bash
hy-smi
```

## 3. 快速开始

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
```

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

请将数据搜索流程所需数据库准备到模型包下的 `data/alphafold3/`。数据搜索流程默认读取的相对结构如下：

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

- AlphaFold3 原始论文：Accurate structure prediction of biomolecular interactions with AlphaFold 3。
- 论文地址：https://www.nature.com/articles/s41586-024-07487-w
- AlphaFold3 源码使用 CC BY-NC-SA 4.0 许可；模型参数受独立使用条款约束。
- 如果在科研工作中使用 AlphaFold3 结果，建议引用 AlphaFold3 原始论文和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。
