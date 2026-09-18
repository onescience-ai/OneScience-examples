<p align="center">
  <strong>
    <span style="font-size: 30px;">Chai-1</span>
  </strong>
</p>

# 模型介绍

Chai-1（chai-lab）是由 Chai Discovery 发布的多模态全原子结构预测模型，可对包含蛋白质、DNA、RNA、小分子配体及糖链的生物分子复合物进行端到端联合结构预测，并输出残基级置信度评分，适用于复合物建模、分子对接与相互作用研究。

论文：Chai-1: Decoding the molecular interactions of life

https://arxiv.org/abs/2502.11047

OneScience 完成了 chai-lab 在 GPU/DCU 上的移植适配与合入，本仓库提供其中的复合物结构预测推理能力。`model/` 目录为自包含的模型层（折叠引擎：特征嵌入、Token 嵌入、trunk 循环、扩散采样与置信度头），不引用环境中安装的 onescience 包；数据管线（datapipes，如输入解析、特征构建、批处理）、评估指标（metrics，候选排序）与输出工具（utils，CIF 写入）由 `scripts/predict.py` 从环境中已安装的 onescience 包导入使用。

# 模型描述

Chai-1 使用 ESM2-3B 提取蛋白质序列嵌入，通过特征嵌入与 Token 嵌入将全原子结构上下文编码为 Token/Atom 表征，经 trunk 循环精炼后，由扩散模块对全原子坐标进行去噪采样，置信度头输出 PAE、PDE 与 pLDDT 评分。多个候选结构按 pTM、ipTM、pLDDT 与冲突（clash）综合分数排序，输出 mmCIF 结构文件。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 复合物结构预测 | 输入 FASTA-like 文件（蛋白/核酸/配体/糖链），输出 CIF 结构与置信度评分 |
| 本地快速验证 | 使用示例输入与 1 次 trunk recycle、20 步扩散快速验证数据读取、模型加载与推理全链路 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |
| 多卡推理 | 通过 `CHAI1_DEVICES=0,1,2,3` 自动将候选数分摊到多卡，最后合并并重新排序 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/chai-lab --local_dir ./chai-lab
cd chai-lab
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装；请使用包含 Chai-1 集成的 onescience 版本
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 权重与数据准备

**权重**

本仓库 `weight/` 目录当前未包含权重文件。请将 Chai-1 模型资产放入 `weight/`，或通过环境变量 `CHAI1_MODEL_DIR` 指定其它位置。目录需要包含：

```text
weight/
├── conformers_v1.apkl
├── esm/
│   └── traced_sdpa_esm2_t36_3B_UR50D_fp16.pt
└── models_v2/
    ├── bond_loss_input_proj.pt
    ├── confidence_head.pt
    ├── diffusion_module.pt
    ├── feature_embedding.pt
    ├── token_embedder.pt
    └── trunk.pt
```

**路径与环境变量**

推理入口的路径与设备由 `CHAI1_*` 环境变量统一控制，也可直接在 `bash scripts/predict.sh` 后追加 Python 脚本支持的参数（多卡模式下模型目录、输出目录、设备、随机种子和采样参数只能通过环境变量设置）：

| 环境变量 | 默认值 | 说明 |
| :---: | :---: | :--- |
| `CHAI1_MODEL_DIR` | `weight` | 模型资产目录（优先于 `ONESCIENCE_MODELS_DIR/chai-lab`） |
| `CHAI1_OUTPUT_DIR` | `outputs/monomer` | 推理输出目录 |
| `CHAI1_INPUT` | `scripts/inputs/example_monomer.fasta` | 输入 FASTA-like 文件 |
| `CHAI1_DEVICE` | `cuda:0` | 单卡推理设备 |
| `CHAI1_PYTHON` | `python` | Python 解释器路径 |
| `CHAI1_SEED` | `42` | 随机种子 |
| `CHAI1_NUM_TRUNK_RECYCLES` | `1` | trunk 循环次数 |
| `CHAI1_NUM_DIFFUSION_TIMESTEPS` | `20` | 扩散去噪步数 |
| `CHAI1_NUM_DIFFUSION_SAMPLES` | `1` | 候选结构数（多卡时为所有卡合计） |
| `CHAI1_NUM_TRUNK_SAMPLES` | `1` | trunk 采样数（多卡模式要求为 1） |

### 推理

脚本默认运行适合验证集成正确性的快速配置（1 次 trunk recycle、20 个扩散步、1 个候选）：

```bash
bash scripts/predict.sh
```

输出写入 `outputs/monomer`，包括预测 CIF（`pred.model_idx_0.cif`）、置信度 NPZ（`scores.model_idx_0.npz`）与按 aggregate score 排序的 `ranking.json`。

使用标准精度配置可设置：

```bash
CHAI1_NUM_TRUNK_RECYCLES=3 \
CHAI1_NUM_DIFFUSION_TIMESTEPS=200 \
CHAI1_NUM_DIFFUSION_SAMPLES=5 \
bash scripts/predict.sh
```

自定义输入文件：

```bash
bash scripts/predict.sh /path/to/input.fasta
```

输入采用 Chai-1 FASTA-like 格式，每个实体必须有唯一名称：

```text
>protein|name=target
MKT...
>ligand|name=ligand
CC(=O)O
>rna|name=rna
AGUC
>dna|name=dna
ATGC
```

本地 MSA、模板命中和约束可分别通过 `--msa-directory`、`--template-hits-path` 和 `--constraint-path` 传入。`--use-msa-server` 与 `--use-templates-server` 会访问外部服务，默认关闭。

### 多卡推理

Chai-1 推理核心是单进程单设备。需要多卡时，在同一计算节点申请多张卡并设置 `CHAI1_DEVICES`，脚本会为每张卡启动独立进程，将总候选数分摊到各卡，最后合并并重新排序结果：

```bash
CHAI1_DEVICES=0,1,2,3 \
CHAI1_NUM_DIFFUSION_SAMPLES=8 \
bash scripts/predict.sh
```

每个进程只看到自己对应的一张卡，因此同时兼容 `CUDA_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES`。

### Python API

完整推理入口为 `scripts/predict.py`（数据管线与候选排序通过环境中已安装的 onescience 包提供）：

```python
import runpy
from pathlib import Path

predict = runpy.run_path("scripts/predict.py")["run_inference"]

candidates = predict(
    fasta_file=Path("input.fasta"),
    output_dir=Path("outputs/example"),
    model_dir=Path("weight"),
    device="cuda:0",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    num_diffn_samples=5,
    seed=42,
)
```

`model/` 包同时导出底层折叠引擎，供高级用户直接调用（输入需为已组装的特征上下文与 collate 后的 batch）：

```python
from pathlib import Path

from model import run_folding_on_context, set_model_dir

set_model_dir(Path("weight"))
# feature_context、batch、bond_ft 由数据管线层（onescience）构建
result = run_folding_on_context(
    feature_context,
    batch,
    bond_ft=bond_ft,
    device="cuda:0",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    num_diffn_samples=5,
    seed=42,
)
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 Chai-1 原始论文的复现与移植版本，模型代码保留原项目 Apache License 2.0 版权头（详见 `model/LICENSE`）。模型、权重及第三方依赖的使用需同时遵守各自许可。

- 如果在科研工作中使用 Chai-1 结果，建议引用：

```bibtex
@article{chai1,
  title = {Chai-1: Decoding the molecular interactions of life},
  author = {Chai Discovery},
  year = {2024},
  url = {https://arxiv.org/abs/2502.11047},
}
```
