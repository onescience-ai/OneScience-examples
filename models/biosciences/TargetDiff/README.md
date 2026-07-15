
<p align="center">
  <strong>
    <span style="font-size: 30px;">TargetDiff</span>
  </strong>
</p>

# 模型介绍

TargetDiff 是用于靶点感知分子生成与蛋白-配体亲和力预测的生物信息模型。模型基于三维等变扩散网络，在给定蛋白结合口袋的条件下生成候选小分子，并可使用 EGNN 属性预测分支对蛋白-配体复合物进行亲和力预测。

原始论文为 _3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction_（ICLR 2023）。

# 仓库说明

本仓库是 OneScience 整理的 TargetDiff 最小可运行模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前包已将 TargetDiff 示例脚本、模型实现快照、样例输入和预训练权重整理到同一个目录。用户在已安装 OneScience 的环境中下载本目录后，可直接在 `targetdiff` 目录下运行 Python 入口脚本。

当前支持能力：

- 使用包内 `3ug2` 蛋白-配体样例和 `egnn_pdbbind_v2016.pt` 权重进行亲和力预测。
- 使用包内 `pretrained_diffusion.pt` 权重对自定义 PDB 口袋进行分子采样。
- 接入 CrossDocked2020 数据后训练 TargetDiff 扩散生成模型。
- 接入 PDBbind 数据后训练、评估蛋白-配体亲和力预测模型。
- 对采样结果进行分子有效性、稳定性和可选 docking 评估。

当前不支持能力：

- 本包不自动安装 OneScience、PyTorch、PyTorch Geometric、RDKit、OpenBabel 等运行依赖。
- 本包不内置 CrossDocked2020、PDBbind 等训练/评估数据集。
- `scripts/*.sh` 中部分脚本保留 OneScience 源仓库路径假设；独立下载本包后，推荐优先使用下文给出的 `python -m scripts...` 命令。

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 蛋白-配体亲和力预测 | 输入蛋白 PDB 与配体 SDF，输出 `Ki`、`Kd` 或 `IC50` 的摩尔浓度预测值 |
| 靶点感知小分子生成 | 输入蛋白结合口袋 PDB，输出生成分子的 `sample.pt` 与可重建分子的 SDF 文件 |
| 扩散模型训练 | 使用 CrossDocked2020 口袋数据训练 TargetDiff 分子生成模型 |
| 属性预测训练 | 使用 PDBbind 数据训练 EGNN 亲和力预测模型 |
| 生成结果评估 | 对采样结果计算稳定性、重建成功率、QED、SA 和可选 Vina docking 指标 |

# 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `examples/3ug2_protein.pdb` | 亲和力预测示例蛋白 | 可直接用于快速推理 |
| `examples/3ug2_ligand.sdf` | 亲和力预测示例配体 | 可直接用于快速推理 |
| `weight/pretrained_diffusion.pt` | TargetDiff 扩散生成模型权重 | 用于分子采样 |
| `weight/egnn_pdbbind_v2016.pt` | EGNN 亲和力预测权重 | 用于亲和力推理 |
| `weight/pk_reg_para.pkl` | 亲和力相关辅助参数文件 | 保留自原始资源 |
| `configs/sampling.yml` | 分子采样配置 | 默认读取环境变量路径，需要按下文建立权重路径映射 |
| `configs/training.yml` | 扩散模型训练配置 | 需要 CrossDocked2020 数据 |
| `configs/prop/*.yml` | 亲和力预测训练配置 | 需要 PDBbind 数据 |
| `scripts/property_prediction/fixed_inference.py` | 亲和力预测推理脚本 | 独立包推荐入口 |
| `scripts/sample_for_pocket.py` | 自定义 PDB 口袋分子采样脚本 | 独立包推荐入口 |
| `scripts/sample_diffusion.py` | 测试集口袋分子采样脚本 | 需要 CrossDocked2020 测试划分 |
| `scripts/evaluate_diffusion.py` | 生成分子评估脚本 | docking 评估需额外安装 Vina/QVina |
| `scripts/train_diffusion.py` | 扩散模型训练脚本 | 需要 CrossDocked2020 数据 |
| `scripts/property_prediction/train_prop.py` | 亲和力预测训练脚本 | 需要 PDBbind 数据 |
| `models/` | TargetDiff 模型实现 | 脚本优先从本目录导入模型；训练日志会复制该目录用于溯源 |

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

请先进入已安装 OneScience 生物信息依赖的环境。

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

### 2. 下载权重、案例包

```bash
bash download_assets.sh
```

### 3. 亲和力预测

#### 3.1 亲和力预测训练

```bash
bash scripts/train_prop.sh
```

该脚本自动完成以下步骤：
1. 从 PDBbind refined set 中提取结合口袋。
2. 按照 coreset 划分训练/验证/测试集。
3. 训练基于 EGNN 的蛋白-配体结合亲和力预测模型。

#### 3.2 亲和力预测评估

使用训练好的亲和力预测模型在测试集上评估，官方在 PDBBind v2016 上的预期指标：

| RMSE | MAE | R² | Pearson | Spearman |
|------|-----|----|---------|----------|
| 1.316 | 1.031 | 0.633 | 0.797 | 0.782 |

运行方式：

```bash
export PYTHONPATH=../../../src:$PYTHONPATH
python scripts/property_prediction/eval_prop.py \
    --ckpt_path ${ONESCIENCE_MODELS_DIR}/targetdiff/pretrained_models/egnn_pdbbind_v2016.pt \
    --device cuda
```

#### 3.3 亲和力预测推理

```bash
bash scripts/inference.sh
```

该脚本默认对示例蛋白-配体对 `3ug2` 进行亲和力预测，使用默认权重和示例数据：

- 模型权重：`${ONESCIENCE_MODELS_DIR}/targetdiff/pretrained_models/egnn_pdbbind_v2016.pt`
- 蛋白：`${ONESCIENCE_DATASETS_DIR}/targetdiff/examples/3ug2_protein.pdb`
- 配体：`${ONESCIENCE_DATASETS_DIR}/targetdiff/examples/3ug2_ligand.sdf`
- 亲和力类型：默认 `Kd`
- 计算设备：默认 `cuda`

### 4. 分子采样

#### 4.1 测试集单条数据采样

```bash
export PYTHONPATH=../../../src:$PYTHONPATH
python -m scripts.sample_diffusion configs/sampling.yml -i 0 --batch_size 50 --result_path ./outputs
```

该脚本从 `configs/sampling.yml` 中读取扩散模型检查点，对测试集第 `i` 个口袋生成候选配体分子，结果保存为 `result_i.pt`。

#### 4.2 多卡批量采样

```bash
bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 0 0
```

参数说明：

| 参数 | 说明 |
|------|------|
| `$1` | 采样配置文件路径 |
| `$2` | 结果输出目录 |
| `$3` | 总工作节点数 |
| `$4` | 当前节点编号（从 0 开始） |
| `$5` | 起始数据索引 |

多卡并行示例：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 0 0 &
CUDA_VISIBLE_DEVICES=1 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 1 0 &
CUDA_VISIBLE_DEVICES=2 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 2 0 &
CUDA_VISIBLE_DEVICES=3 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 3 0 &
wait
```

脚本内部固定 `TOTAL_TASKS=100`、`BATCH_SIZE=50`，按索引取模分配给各节点。

#### 4.3 自定义 PDB 口袋采样

```bash
export PYTHONPATH=../../../src:$PYTHONPATH
python -m scripts.sample_for_pocket configs/sampling.yml \
    --pdb_path /path/to/pocket.pdb \
    --result_path ./outputs_pdb \
    --num_samples 5 \
    --batch_size 1
```
参数说明：

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `config` | 是 | 采样配置文件路径 |
| `--pdb_path` | 是 | 蛋白口袋 PDB 文件路径（建议为 10Å 口袋） |
| `--result_path` | 否 | 结果输出目录，默认 `./outputs_pdb` |
| `--num_samples` | 否 | 采样分子数量，默认读取配置文件 |
| `--batch_size` | 否 | 批量大小，默认 `100` |
| `--device` | 否 | 计算设备，默认 `cuda:0` |

采样结果会保存为 `outputs_pdb/sample.pt`，成功重建的分子会额外输出到 `outputs_pdb/sdf/`。


### 5. 生成分子评估
#### 5.1 从采样结果评估

若需对接评估（`vina_score` / `vina_dock` / `qvina`），还需安装：

```bash
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

```bash
export PYTHONPATH=../../../src:$PYTHONPATH
python scripts/evaluate_diffusion.py ./outputs --docking_mode vina_score --protein_root /path/to/protein_root
```

参数说明：

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `sample_path` | 是 | 采样结果目录，包含 `result_*.pt` 文件 |
| `--docking_mode` | 是 | 对接模式，可选 `none`、`vina_score`、`vina_dock`、`qvina` |
| `--protein_root` | 否 | 原始蛋白文件根目录，用于对接评估 |
| `--eval_step` | 否 | 评估第几步的采样结果，默认 `-1`（最后一步） |
| `--eval_num_examples` | 否 | 评估样本数量，默认全部 |
| `--exhaustiveness` | 否 | 对接搜索强度，默认 `16` |
| `--save` | 否 | 是否保存评估结果，默认 `True` |

支持的对接模式：

| 模式 | 说明 |
|------|------|
| `none` | 仅计算 validity、uniqueness、novelty 等指标，不进行对接 |
| `vina_score` | 使用 AutoDock Vina 对生成分子进行打分 |
| `vina_dock` | 使用 AutoDock Vina 对生成分子进行重新对接 |
| `qvina` | 使用 QuickVina 进行对接 |

首次运行 `vina_score` 或 `vina_dock` 模式时，需要一定时间准备 `pdbqt` 和 `pqr` 文件。

#### 5.2 从 meta 文件评估

官方提供了已采样并对接好的 meta 文件（包含 TargetDiff 及 CVAE、AR、Pocket2Mol 等基线），可直接下载评估：

| Meta 文件 | 对应论文 |
|-----------|----------|
| `crossdocked_test_vina_docked.pt` | 原始测试集对接结果 |
| `cvae_vina_docked.pt` | liGAN |
| `ar_vina_docked.pt` | AR |
| `pocket2mol_vina_docked.pt` | Pocket2Mol |
| `targetdiff_vina_docked.pt` | TargetDiff |

官方 meta 文件下载地址：https://drive.google.com/drive/folders/19imu-mlwrjnQhgbXpwsLgA17s1Rv70YS?usp=share_link

评估命令：

```bash
export PYTHONPATH=../../../src:$PYTHONPATH
python scripts/evaluate_from_meta.py sampling_results/targetdiff_vina_docked.pt --result_path eval_targetdiff
```

参数说明：

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `meta_file` | 是 | 包含采样与对接结果的 `.pt` 文件 |
| `--result_path` | 否 | 评估结果输出目录，默认 `eval_results` |

---

### 6. 扩散模型训练

```bash
bash train_diffusion.sh
```

默认读取 `configs/training.yml`，在 CrossDocked2020 口袋数据集上训练，日志与检查点保存在 `./logs_diffusion/`。

**自定义配置或覆盖参数：**

```bash
# 指定自定义配置文件
bash train_diffusion.sh configs/custom_training.yml

# 命令行覆盖训练参数
bash train_diffusion.sh --train.batch_size 8 --train.max_iters 500000
```

**主要训练参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.path` | `${ONESCIENCE_DATASETS_DIR}/targetdiff/data/crossdocked_v1.1_rmsd1.0_pocket10` | 训练数据目录 |
| `data.split` | `${ONESCIENCE_DATASETS_DIR}/targetdiff/data/crossdocked_pocket10_pose_split.pt` | 训练/验证/测试划分文件 |
| `train.batch_size` | `4` | 批次大小 |
| `train.max_iters` | `10000000` | 最大迭代次数 |
| `train.lr` | `5.e-4` | 学习率 |
| `logdir` | `./logs_diffusion` | 日志输出目录 |

---
### 7. 数据预处理

#### 7.1 CrossDocked2020 数据预处理

如需从头处理 CrossDocked2020 数据，按以下步骤执行：

1. 下载 CrossDocked2020 v1.1 并保存到 `data/CrossDocked2020`。
2. 过滤 RMSD < 1Å 的样本：

    ```bash
    export PYTHONPATH=../../../src:$PYTHONPATH
    python scripts/data_preparation/clean_crossdocked.py \
        --source data/CrossDocked2020 \
        --dest data/crossdocked_v1.1_rmsd1.0 \
        --rmsd_thr 1.0
    ```

3. 从蛋白中提取 10Å 结合口袋：

    ```bash
    python scripts/data_preparation/extract_pockets.py \
        --source data/crossdocked_v1.1_rmsd1.0 \
        --dest data/crossdocked_v1.1_rmsd1.0_pocket10
    ```

4. 划分训练集与测试集：

    ```bash
    python scripts/data_preparation/split_pl_dataset.py \
        --path data/crossdocked_v1.1_rmsd1.0_pocket10 \
        --dest data/crossdocked_pocket10_pose_split.pt \
        --fixed_split data/split_by_name.pt
    ```

#### 7.2 PDBbind 数据预处理

亲和力预测训练脚本 `train_prop.sh` 已自动完成口袋提取与数据集划分。如需单独执行，可使用以下命令：

```bash
export PYTHONPATH=../../../src:$PYTHONPATH

python scripts/property_prediction/extract_pockets.py \
    --source data/pdbbind_v2020 \
    --dest data/pdbbind_v2020_processed \
    --subset refined \
    --num_workers 16

python scripts/property_prediction/pdbbind_split.py \
    --split_mode coreset \
    --index_path data/pdbbind_v2020_processed/pocket_10_refined/index.pkl \
    --test_path data/pdbbind_v2016/coreset \
    --save_path data/pdbbind_v2020_processed/pocket_10_refined/split.pt
```

---

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- TargetDiff 原始代码使用 MIT License。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。
- 如果在科研工作中使用 TargetDiff 结果，建议引用 TargetDiff 原始论文和 OneScience 相关项目信息，并根据实际任务补充 CrossDocked2020、PDBbind、RDKit、OpenBabel、Vina/QVina 等数据集或工具引用。

```bibtex
@inproceedings{guan3d,
  title={3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction},
  author={Guan, Jiaqi and Qian, Wesley Wei and Peng, Xingang and Su, Yufeng and Peng, Jian and Ma, Jianzhu},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
