
<p align="center">
  <strong>
    <span style="font-size: 30px;">DiffDock</span>
  </strong>
</p>

# 模型介绍

DiffDock 是由 Corso 等人提出的蛋白-配体分子对接扩散模型。模型将 docking 视为生成式建模问题，在给定蛋白受体和小分子配体后，通过 SO(3)/SE(3) 上的扩散过程同时建模配体的平移、旋转和扭转自由度，生成候选结合构象。

本包整理的是 OneScience 中 DiffDock/DiffDock-L 的 CGModel 主路径，包含训练、采样和评估入口。

# 仓库说明

本仓库是 OneScience 整理的 DiffDock 最小可运行模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前包已将 DiffDock 示例脚本、模型实现快照、样例蛋白-配体输入和配置文件整理到同一个目录。用户在已安装 OneScience 生物信息依赖的环境中下载本目录后，可直接在 `diffdock` 目录下运行 Python 入口脚本。

当前支持能力：

- 接入 PDBBind 数据后训练 DiffDock score 模型。
- 接入 MOAD 数据后训练 DiffDock score 模型。
- 使用用户提供的 score 模型权重进行单复合物或 CSV 批量分子对接采样。
- 使用用户提供的 score/confidence 模型权重进行 confidence rerank 采样。
- 对数据集采样结果进行评估，可选接入 GNINA 能量最小化。
- 使用 OneScience 包内 confidence 训练模块训练 confidence 模型。

当前不支持能力：

- 本包不内置 DiffDock 预训练权重；因此下载后不能直接进行无需权重的推理，只能先训练，或手动提供已训练的 score/confidence checkpoint。
- 本包不自动安装 OneScience、PyTorch、PyTorch Geometric、RDKit、OpenBabel、e3nn、torch-scatter、torch-cluster 等运行依赖。
- 本包不内置 PDBBind、MOAD、DockGen、PoseBusters 等训练或评估数据集。
- 当前迁移主路径为 CGModel；`all_atoms=true`、旧 CG 模型、`dataset=pdbsidechain`、`dataset=distillation`、`triple_training=true` 等路径未作为独立包推荐入口。
- `scripts/*.sh` 保留了部分 OneScience 源仓库环境假设；独立下载本包后，推荐优先使用下文 `python -m scripts...` 命令。

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| Score 模型训练 | 输入处理后的 PDBBind 或 MOAD 数据，输出 DiffDock score 模型 checkpoint |
| 单复合物分子对接 | 输入蛋白 PDB 与配体 SMILES/SDF/MOL2，输出候选配体结合构象 SDF |
| 批量分子对接 | 输入包含蛋白、配体和复合物名称的 CSV，批量输出采样构象 |
| Confidence rerank | 使用额外 confidence 模型对采样构象排序 |
| 数据集评估 | 对验证集或测试集采样结果计算 RMSD 等指标，可选调用 GNINA |

# 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `data/6o5u_protein_processed.pdb` | 示例蛋白结构 | 可用于采样配置示例 |
| `data/6o5u_ligand.sdf` | 示例配体结构 | 可用于采样配置示例 |
| `data/1a46_protein_processed.pdb` | 示例蛋白结构 | 可用于自定义测试 |
| `data/1a46_ligand.sdf` | 示例配体结构 | 可用于自定义测试 |
| `configs/training.yml` | score 模型训练配置 | 需要改成实际数据集路径 |
| `configs/sampling.yml` | 分子对接采样配置 | 需要提供 score/confidence 权重路径 |
| `configs/evaluate.yml` | 数据集评估配置 | 需要提供数据、模型和输出路径 |
| `scripts/train_diffdock.py` | score 模型训练入口 | 推荐使用 `python -m scripts.train_diffdock` |
| `scripts/sample_diffdock.py` | 单复合物或 CSV 批量采样入口 | 推荐使用 `python -m scripts.sample_diffdock` |
| `scripts/evaluate.py` | 数据集评估入口 | 调用 OneScience 中的 DiffDock 评估模块 |
| `scripts/train.sh` | 训练 shell 示例 | 保留源仓库环境变量假设 |
| `scripts/infer.sh` | 采样 shell 示例 | 保留源仓库环境变量假设 |
| `scripts/train_demo.sh` | smoke 训练 shell 示例 | 需要本地数据路径可用 |
| `scripts/*.sbatch` | Slurm 作业模板 | 需要按集群环境修改 |
| `models/` | DiffDock 模型实现快照 | 包内脚本已改为优先使用该目录下的本地模型代码 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于连通性验证，但速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**软件要求**

DiffDock 常用依赖包括 PyTorch、PyTorch Geometric、RDKit、OpenBabel、e3nn、torch-scatter、torch-cluster、NumPy、SciPy、tqdm、PyYAML 等。数据集评估中的 GNINA 能量最小化需要额外安装 `gnina`。

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光 DCU：

```bash
hy-smi
```

## 快速开始

### 1. 安装onescience库

```bash
git clone https://gitee.com/onescience-ai/onescience
cd onescience
bash install.sh bio
```

### 2. 下载数据库(待上传)

```bash
modelscope download --dataset OneScience/diffdock --local_dir ./diffdock
cd diffdock
```

DiffDock 训练通常需要预处理后的 PDBBind 或 MOAD 数据。推荐将数据组织到统一根目录，例如：

```text
${ONESCIENCE_DATASETS_DIR}/diffdock/
├── PDBBind_processed/
├── MOAD_processed/
└── splits/
    ├── timesplit_no_lig_overlap_train
    ├── timesplit_no_lig_overlap_val
    └── timesplit_test
```

划分文件为纯文本格式，每行一个复合物名称，需与数据目录中的子目录名对应。

### 3. 检查包内文件

当前包不包含预训练权重，下面命令应只看到源码、配置和示例输入文件：

```bash
find . -maxdepth 3 -type f
```

若需要采样或评估，请先通过训练得到 checkpoint，或将外部 DiffDock score/confidence 权重放到本地目录，并保证权重目录中存在 `model_parameters.yml`。

### 4. Score 模型训练

训练前请将 `configs/training.yml` 中的数据路径改为本机路径，常用字段包括：

- `data.pdbbind_dir`：处理后的 PDBBind 数据目录。
- `data.moad_dir`：处理后的 MOAD 数据目录。
- `data.split_train`：训练集复合物列表。
- `data.split_val`：验证集复合物列表。
- `runtime.log_dir`：训练输出目录。

启动训练：

```bash
cd scripts
bash train.sh
```

训练成功后，输出目录通常包含：

- `model_parameters.yml`：模型结构与训练参数。
- `best_model.pt`：验证损失最优权重。
- `best_ema_model.pt`：EMA 权重。
- `best_inference_epoch_model.pt` 或 `best_ema_inference_epoch_model.pt`：开启 inference validation 时保存的权重。
- `last_model.pt`：最后一个 epoch 的训练状态。

### 5. 分子对接采样

采样需要一个已训练好的 score 模型目录，例如：

```text
outputs/train/diffdock_cg_example/
├── model_parameters.yml
└── best_model.pt
```

请将 `configs/sampling.yml` 中的关键字段改为实际路径：

- `model.model_dir`：score 模型目录。
- `model.ckpt`：score checkpoint 文件名。
- `confidence.confidence_model_dir`：confidence 模型目录；不使用 rerank 时设为 `null`。
- `input.protein_path`：蛋白 PDB 路径，可使用 `data/6o5u_protein_processed.pdb`。
- `input.ligand_description`：SMILES 字符串或配体 SDF/MOL2 路径，可使用 `data/6o5u_ligand.sdf`。
- `runtime.out_dir`：采样输出目录。

启动单复合物采样：

```bash
cd scripts
bash infer.sh
```

### 6. CSV 批量采样

将 `configs/sampling.yml` 中 `input.protein_ligand_csv` 设置为 CSV 路径，并将单复合物字段按需设为 `null`。CSV 建议包含以下列：

```text
complex_name,protein_path,ligand_description,protein_sequence
```

其中 `ligand_description` 可以是 SMILES，也可以是 SDF/MOL2 文件路径。

### 7. 数据集评估

评估前请将 `configs/evaluate.yml` 中的数据集、模型和输出目录改成实际路径：

```bash
python -m scripts.evaluate --config configs/evaluate.yml
```

若启用 `gnina.gnina_minimize=true`，请先确认当前环境中可以直接执行：

```bash
gnina --help
```

### 8. Confidence 模型训练

confidence 训练入口位于已安装的 OneScience 包中，本 ModelScope 包不重复复制该模块。完成 score 模型训练后，可在 OneScience 环境中运行：

```bash
python -m onescience.confidence.diffdock.confidence_train \
    --original_model_dir outputs/train/diffdock_cg_example \
    --data_dir /path/to/PDBBind_processed \
    --split_train /path/to/splits/timesplit_no_lig_overlap_train \
    --split_val /path/to/splits/timesplit_no_lig_overlap_val
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- DiffDock 原始代码使用 MIT License。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。
- 如果在科研工作中使用 DiffDock 结果，建议引用 DiffDock、DiffDock-L 原始论文和 OneScience 相关项目信息，并根据实际任务补充 PDBBind、MOAD、RDKit、OpenBabel、GNINA 等数据集或工具引用。

```bibtex
@inproceedings{corso2023diffdock,
  title={DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking},
  author={Corso, Gabriele and St{\"a}rk, Hannes and Jing, Bowen and Barzilay, Regina and Jaakkola, Tommi},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
