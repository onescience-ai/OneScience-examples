<p align="center">
  <strong>
    <span style="font-size: 30px;">ESM</span>
  </strong>
</p>

# 模型介绍

ESM（Evolutionary Scale Modeling）是 Meta AI / FAIR 发布的蛋白质语言模型家族，面向蛋白质序列表征、结构预测、变异效应评估和固定骨架序列设计等任务。本项目包含 ESM-1、ESM-2、MSA Transformer、ESMFold、ESM-1v 和 ESM-IF1 相关能力，可支持从 FASTA 提取蛋白表征、预测 PDB 结构、对 DMS 突变数据进行零样本评分，以及基于已知结构进行 inverse folding 采样和序列打分。

主要论文：

- ESM-2 / ESMFold：Evolutionary-scale prediction of atomic-level protein structure with a language model  
  https://www.science.org/doi/10.1126/science.ade2574
- ESM 系列基础模型：Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences  
  https://www.pnas.org/doi/10.1073/pnas.2016239118
- MSA Transformer：MSA Transformer  
  https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1
- ESM-1v 变异效应预测：Language models enable zero-shot prediction of the effects of mutations on protein function  
  https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1
- ESM-IF1 inverse folding：Learning inverse folding from millions of predicted structures  
  https://www.biorxiv.org/content/early/2022/04/10/2022.04.10.487779

# 仓库说明

本仓库是 ESM 最小可运行独立模型仓库，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- ESM / ESM-2 蛋白序列表征提取
- ESMFold 单序列蛋白三维结构预测
- ESM-IF1 固定骨架 inverse folding 序列采样
- ESM-IF1 基于结构的序列 log-likelihood 打分
- ESM-1v / MSA Transformer 零样本变异效应预测
- 小样例 FASTA、PDB / CIF 和 DMS 数据的本地流程验证

当前不支持能力：

- 不负责外部 MSA 搜索、结构数据库检索或蛋白结构可视化服务
- 不面向临床诊断或医学决策

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 蛋白序列表征提取 | 输入 FASTA 文件，输出 per-token、mean、BOS 或 contact 表征 |
| 蛋白结构预测 | 输入一条或多条氨基酸序列，输出对应 PDB 结构文件 |
| 变异效应评分 | 输入野生型序列和 DMS 突变表，输出突变影响评分 |
| 固定骨架序列设计 | 输入 PDB / CIF 结构和链 ID，采样满足骨架约束的候选序列 |
| 结构条件序列打分 | 输入结构和候选序列，计算 conditional log-likelihood |
| OneCode / 本地运行 | 在生物领域运行环境中快速验证脚本连通性 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `model/esm/` | ESM 模型源码 | 包含 ESM-1、ESM-2、MSA Transformer、ESMFold 和 inverse folding |
| `model/openfold/` | ESMFold 结构模块依赖 | 用于结构预测 |
| `model/protenix/layer_norm/` | 可选 LayerNorm 加速模块 | GPU 环境可按需使用 |
| `scripts/extract.py` | 序列表征提取脚本 | 读取 FASTA 并保存 `.pt` 表征文件 |
| `scripts/fold.py` | ESMFold 结构预测脚本 | 输出 PDB 文件 |
| `scripts/infer.sh` | 默认连通性示例脚本 | 默认执行 ESM-2 表征提取 |
| `scripts/inverse_folding/sample_sequences.py` | inverse folding 采样脚本 | 根据结构采样蛋白序列 |
| `scripts/inverse_folding/score_log_likelihoods.py` | inverse folding 序列打分脚本 | 根据结构评估候选序列 |
| `scripts/variant_prediction/predict.py` | 变异效应预测脚本 | 支持 ESM-1v 和 MSA Transformer |
| `scripts/check_import_boundaries.py` | 静态导入检查脚本 | 用于工程完整性验证 |
| `data/fasta/` | FASTA 样例数据 | 用于表征提取和结构预测示例 |
| `data/inverse_folding/` | PDB / CIF 和候选序列样例 | 用于 inverse folding 示例 |
| `data/variant_prediction/` | DMS 变异效应预测样例数据 | 用于变异评分示例 |
| `weight/` | 权重占位目录 | 建议放置 `weight/checkpoints/*.pt` |
| `tests/` | 静态测试 | 验证导入边界 |
| `LICENSE` | 开源许可证 | MIT License |

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

本模型包已包含少量样例数据，可直接用于默认流程验证。

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```
```bash
#如果需要找不到库的情况需要激活cuda，参考下列代码
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
```


安装完成后回到模型包目录：

```bash
cd ./ESM
```

### 准备权重

请将所需 ESM 权重放置在以下目录：

```text
weight/
  checkpoints/
    esm2_t6_8M_UR50D.pt
    esmfold_3B_v1.pt
    esm1v_t33_650M_UR90S_1.pt
    esm_if1_gvp4_t16_142M_UR50.pt
    ...
```

如果使用共享运行目录，也可以通过环境变量指定权重位置：

```bash
export ESM_WEIGHT_DIR=/path/to/esm/weight
```

默认示例会读取：

- `weight/checkpoints/esm2_t6_8M_UR50D.pt`

ESMFold、ESM-1v 和 ESM-IF1 示例需要对应权重可用。

### 默认示例

```bash
bash scripts/infer.sh
```

默认示例会读取 `data/fasta/few_proteins.fasta`，使用 `esm2_t6_8M_UR50D.pt` 提取蛋白表征，并将结果保存至 `outputs/embeddings/`。

### 序列表征提取

```bash
python scripts/extract.py \
  weight/checkpoints/esm2_t6_8M_UR50D.pt \
  data/fasta/few_proteins.fasta \
  outputs/embeddings \
  --include mean per_tok \
  --repr_layers 6
```

### ESMFold 结构预测

```bash
python scripts/fold.py \
  -i data/fasta/few_proteins.fasta \
  -o outputs/pdb \
  --model-dir weight \
  --cpu-only
```

输出目录会生成一个或多个 `.pdb` 文件。正式 GPU / DCU 推理时，可去掉 `--cpu-only`，并根据显存情况设置 `--chunk-size` 或 `--max-tokens-per-batch`。

也可以通过默认脚本显式启用 ESMFold：

```bash
RUN_ESMFOLD=1 bash scripts/infer.sh
```

### inverse folding 序列采样

```bash
python scripts/inverse_folding/sample_sequences.py \
  data/inverse_folding/5YH2.pdb \
  --chain A \
  --outpath outputs/sampled_seqs.fasta \
  --num-samples 1 \
  --nogpu
```

### inverse folding 序列打分

```bash
python scripts/inverse_folding/score_log_likelihoods.py \
  data/inverse_folding/5YH2.pdb \
  data/inverse_folding/5YH2_mutated_seqs.fasta \
  --chain A \
  --outpath outputs/sequence_scores.csv \
  --nogpu
```

### 变异效应预测

变异效应预测需要提供与 DMS 突变列匹配的野生型序列：

```bash
python scripts/variant_prediction/predict.py \
  --model-location esm1v_t33_650M_UR90S_1 \
  --sequence "${ESM_VARIANT_SEQUENCE}" \
  --dms-input data/variant_prediction/BLAT_ECOLX_Ranganathan2015.csv \
  --mutation-col mutant \
  --dms-output outputs/variant_prediction.csv \
  --offset-idx 24 \
  --scoring-strategy wt-marginals
```

# 数据格式

样例数据默认存放在 `data/` 下：

```text
data/
  fasta/
    few_proteins.fasta
    some_proteins.fasta
  inverse_folding/
    5YH2.pdb
    5YH2.cif
    5YH2_mutated_seqs.fasta
    example.json
  variant_prediction/
    BLAT_ECOLX_Ranganathan2015.csv
    rho_pp.csv
    aggregated_rho.csv
    aggregated_rho_round3.csv
```

其中：

- FASTA 文件用于序列表征提取和结构预测。
- PDB / CIF 文件用于 inverse folding 采样和结构条件序列打分。
- 变异效应预测 CSV 需包含突变列，默认示例列名为 `mutant`，突变格式形如 `A123B`。
- 自定义 DMS 数据需要保证 `--sequence` 提供的野生型序列与突变列中的野生型氨基酸一致。

# 验证

静态导入检查：

```bash
python scripts/check_import_boundaries.py
```

语法检查：

```bash
python -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for root in ['model', 'scripts', 'tests'] for p in pathlib.Path(root).rglob('*.py')]"
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- ESM-2 / ESMFold 原始论文：Evolutionary-scale prediction of atomic-level protein structure with a language model。
- ESM 系列基础模型论文：Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences。
- MSA Transformer 原始论文：MSA Transformer。
- ESM-1v 原始论文：Language models enable zero-shot prediction of the effects of mutations on protein function。
- ESM-IF1 原始论文：Learning inverse folding from millions of predicted structures。
- ESM 相关源码使用 MIT License，见 `LICENSE`。模型权重和数据的使用条款请以对应发布方说明为准。
- 如果在科研工作中使用 ESM 结果，建议引用对应 ESM 原始论文和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。
