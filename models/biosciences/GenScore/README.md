
<p align="center">
  <strong>
    <span style="font-size: 30px;">GenScore</span>
  </strong>
</p>

# 模型介绍

GenScore 是一个基于图神经网络的**蛋白质-配体打分框架**，由 RTMScore 扩展而来。它能够预测蛋白-小分子结合亲和力并评估对接构象质量，在多个数据集上展现出均衡的打分（scoring）、排序（ranking）、对接（docking）和虚拟筛选（screening）能力。

# 仓库说明

本示例将 GenScore 集成到 OneScience 生物信息（AI for Biology）组件中，提供蛋白-配体打分、口袋生成、贡献度分析、模型训练以及 CASF-2016 基准评测的统一入口。

当前支持能力：

- **蛋白-配体打分**：对给定蛋白（或已提取的结合口袋）与配体构象预测结合分数。
- **口袋自动生成**：基于参考配体位置从完整蛋白结构中自动截取结合口袋。
- **贡献度分析**：输出原子级别和残基级别对最终打分值的贡献，辅助可解释性分析。
- **模型训练**：基于 PDBbind 预处理后的蛋白-配体图数据训练 GenScore 打分网络。
- **CASF-2016 基准评测**：支持打分/排序（scoring/ranking）、对接（docking）和虚拟筛选（screening）三项标准测试。

当前不支持能力：

- 当前不存在预训练权重及训练数据集，可通过modelscope下载相应dataset，并通过export指定路径使用。

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 蛋白-配体打分 | 对给定蛋白（或已提取的结合口袋）与配体构象预测结合分数。 |
| 口袋自动生成 | 基于参考配体位置从完整蛋白结构中自动截取结合口袋。 |
| 贡献度分析 | 输出原子级别和残基级别对最终打分值的贡献，辅助可解释性分析。 |
| 模型训练 | 基于 PDBbind 预处理后的蛋白-配体图数据训练 GenScore 打分网络。 |
| CASF-2016 基准评测 | 支持打分/排序（scoring/ranking）、对接（docking）和虚拟筛选（screening）三项标准测试。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 本文档 | 中文说明文档 |
| `run_genscore.sh` | 蛋白-配体打分推理示例脚本 | 已提供 |
| `train_genscore_smoke.sh` | 小规模训练冒烟测试脚本 | 已提供 |
| `train_genscore_full.sh` | 完整训练脚本 | 已提供 |
| `run_genscore_benchmarks.sh` | CASF-2016 基准测试脚本 | 已提供 |
| `genscore.py` | 推理入口 | - |
| `train_genscore.py` | 训练入口 | - |
| `preprocess_pdbbind.py` | PDBbind 数据预处理入口 | - |
| `benchmarks/` | CASF-2016 评测脚本 | 包含 docking、scoring/ranking 和 screening 脚本 |
| `benchmark_data/` | 评测辅助数据 | - |

```
├── benchmarks/                       # CASF-2016 评测脚本
│   ├── casf2016_docking.py
│   ├── casf2016_scoring_ranking.py
│   └── casf2016_screening.py
├── models/                           # 模型实现
├── scripts/                           # 运行入口
    ├── run_genscore.sh                   # 蛋白-配体打分推理示例脚本（已提供）
    ├── train_genscore_smoke.sh           # 小规模训练冒烟测试脚本（已提供）
    ├── train_genscore_full.sh            # 完整训练脚本（已提供）
    ├── run_genscore_benchmarks.sh        # CASF-2016 基准测试脚本（已提供）
    ├── genscore.py                       # 推理入口
    ├── train_genscore.py                 # 训练入口
    ├── preprocess_pdbbind.py             # PDBbind 数据预处理入口
├── benchmark_data/                   # 评测辅助数据
└── README.md                         # 本文档
```

# 使用说明

## 1. OneCode使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用GPU或DCU运行。
- CPU可以用于连通性验证，但速度较慢。
- DCU用户需要预先安装DTK，建议使用DTK 25.04.2以上版本或与当前集群匹配的OneScience推荐版本。

**软件要求**

DCU用户想了解更多适配内容请联系 liubiao@sugon.com

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光DCU：

```bash
hy-smi
```

## 快速开始

### 1. 安装运行环境

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```
#如果下述代码运行存在找不到库的情况，需要激活cuda，参考下列代码

```bash
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
```

### 2. 下载数据库(含权重)

```bash
modelscope download --dataset OneScience/GenScore --local_dir ./dataset
cd model
```

### 3. 使用方式

#### 1. 蛋白-配体打分推理（`run_genscore.sh`）

```bash
cd ./scripts
bash run_genscore.sh
```

该脚本依次运行 4 个推理示例，覆盖常见使用场景：

1. **GT 模型 + 自动生成口袋**：输入完整蛋白、参考配体与 decoy 配体，自动生成结合口袋后打分。
2. **GatedGCN 模型 + 预提取口袋**：输入已预提取的口袋 PDB 与 decoy 配体进行打分。
3. **原子贡献分析**：使用 GatedGCN 模型计算每个原子对打分的贡献。
4. **残基贡献分析**：使用 GatedGCN 模型计算每个残基对打分的贡献。

#### 2. 模型训练

##### 2.1 小规模冒烟测试（`train_genscore_smoke.sh`）

```bash
cd ./scripts
bash train_genscore_smoke.sh
```

默认配置：

| 参数 | 默认值 |
|------|--------|
| 训练轮数 | 100 |
| 批次大小 | 16 |
| 验证集样本数 | 1500 |
| 编码器 | `gatedgcn` |
| 输出模型 | `genscore_smoke_bs16.pth` |

##### 2.2 完整训练（`train_genscore_full.sh`）

```bash
cd ./scripts
bash train_genscore_full.sh
```

支持通过环境变量覆盖关键参数：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `GENSCORE_DATA_DIR` | 训练数据目录 | `${ONESCIENCE_DATASETS_DIR}/GenScore/genscore_data/rtmscore_s` |
| `GENSCORE_DATA_PREFIX` | 数据文件前缀 | `v2020_train` |
| `GENSCORE_ENCODER` | 图编码器类型 | `gatedgcn` |
| `GENSCORE_MODEL_PATH` | 输出模型路径 | `examples/biosciences/genscore/genscore_${GENSCORE_ENCODER}_full_3000.pth` |
| `GENSCORE_NUM_EPOCHS` | 训练轮数 | `3000` |
| `GENSCORE_BATCH_SIZE` | 批次大小 | `64` |
| `GENSCORE_NUM_WORKERS` | 数据加载线程数 | `8` |
| `GENSCORE_VALNUM` | 验证集样本数 | `1500` |
| `GENSCORE_PATIENCE` | 早停耐心值 | `70` |
---

#### 3. CASF-2016 基准评测（`run_genscore_benchmarks.sh`）

```bash
cd ./scripts
bash run_genscore_benchmarks.sh all
```

可单独运行某一项评测任务：

```bash
bash run_genscore_benchmarks.sh scoring
bash run_genscore_benchmarks.sh docking
bash run_genscore_benchmarks.sh screening
```

评测任务说明：

| 任务 | 说明 |
|------|------|
| `scoring` | 打分能力评测（scoring/ranking） |
| `docking` | 对接能力评测（docking power） |
| `screening` | 虚拟筛选能力评测（screening power） |

### 数据预处理

训练需要预处理后的 PDBbind 图数据，包括：

```
<data_prefix>_ids.npy
<data_prefix>_lig.pt
<data_prefix>_prot.pt
```

可通过以下命令对原始 PDBbind 数据进行预处理：

```bash
cd ./scripts
export PYTHONPATH=../../../src:$PYTHONPATH
python preprocess_pdbbind.py \
  --dir /path/to/pdbbind \
  --ref /path/to/pdbbind_2020_general.csv \
  --cutoff 10.0 \
  --outprefix /path/to/preprocessed/pdbbind/v2020_train
```

主要参数：

| 参数 | 说明 |
|------|------|
| `--dir` | PDBbind 原始数据目录 |
| `--ref` | PDBbind 索引 CSV 文件 |
| `--cutoff` | 蛋白-配体距离截断，默认 `10.0` Å |
| `--outprefix` | 输出文件前缀 |

### 注意事项

- 运行脚本前需确保 `ONESCIENCE_DATASETS_DIR` 环境变量已正确设置。
- 脚本会自动设置 ROCm/DCU 相关的 `LD_LIBRARY_PATH`，在海光 DCU 平台可直接运行。
- 自动生成口袋依赖 OpenBabel 和 ProDy；如 OpenBabel 需要显式数据路径，请提前设置 `BABEL_LIBDIR` 和 `BABEL_DATADIR`。
- 训练与推理默认使用单卡（`HIP_VISIBLE_DEVICES=0` 或 `CUDA_VISIBLE_DEVICES=0`），多卡训练请调整训练脚本中的并行策略。
- 训练脚本会检查 `<data_prefix>_ids.npy`、`<data_prefix>_lig.pt`、`<data_prefix>_prot.pt` 是否存在，缺失会报错退出。
- CASF-2016 评测中 `docking` 和 `screening` 任务需要完整的 CASF-2016 与 PDBbind v2020 数据。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

如果您在研究中使用了 GenScore，请引用原始工作：

```bibtex
@article{genScore,
  title={GenScore: a generalized protein-ligand scoring framework},
  author={Shen, Chao and Hu, Yafeng and Wang, Zhe and Zhang, Xujun and Li, Jianxin and Wang, Guisheng and Wang, Tingjun and Chen, Yen-Wei and Pan, Peichen and Hou, Tingjun},
  journal={Journal of Chemical Information and Modeling},
  year={2023}
}
```

更多信息请参考 GenScore 官方仓库：https://github.com/sc8668/GenScore

许可证信息请以 [GenScore 官方仓库](https://github.com/sc8668/GenScore) 中的说明为准。
