
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

### 1. 安装onescience库

```bash
git clone https://gitee.com/onescience-ai/onescience
cd onescience
bash install.sh bio
```

### 2. 下载权重、案例包&激活环境

```bash
bash download_assets.sh

#如果需要找不到库的情况需要激活cuda，参考下列代码
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"

```

### 3. 亲和力预测示例

该示例只使用包内已有的权重和 `3ug2` 输入文件，是下载后最小可运行的验证入口。

```bash
python -m scripts.property_prediction.fixed_inference \
    --ckpt_path weight/egnn_pdbbind_v2016.pt \
    --protein_path examples/3ug2_protein.pdb \
    --ligand_path examples/3ug2_ligand.sdf \
    --kind Kd \
    --device cuda
```

CPU 连通性验证可运行：

```bash
python -m scripts.property_prediction.fixed_inference \
    --ckpt_path weight/egnn_pdbbind_v2016.pt \
    --protein_path examples/3ug2_protein.pdb \
    --ligand_path examples/3ug2_ligand.sdf \
    --kind Kd \
    --device cpu
```

运行成功后，控制台会输出类似如下结果：

```text
PDB ID: examples/3ug2_protein.pdb Prediction: Kd=...e-.. m
```

### 4. 自定义口袋分子采样

`configs/sampling.yml` 沿用了 OneScience 数据目录变量，默认读取：

```text
${ONESCIENCE_DATASETS_DIR}/targetdiff/pretrained_models/pretrained_diffusion.pt
```

独立下载本包后，可在 `targetdiff` 目录下建立一次权重路径映射：

```bash
mkdir -p pretrained_models
ln -sf ../weight/pretrained_diffusion.pt pretrained_models/pretrained_diffusion.pt
export ONESCIENCE_DATASETS_DIR="$(dirname "$PWD")"
```

随后可对 PDB 口袋采样。建议输入已经截取好的结合口袋 PDB；直接输入完整蛋白可用于连通性验证，但结果和速度不作为推荐设置。

```bash
python -m scripts.sample_for_pocket configs/sampling.yml \
    --pdb_path examples/3ug2_protein.pdb \
    --result_path outputs_pdb \
    --num_samples 16 \
    --batch_size 16 \
    --device cuda:0
```

输出文件：

- `outputs_pdb/sample.pt`：采样轨迹与生成结果。
- `outputs_pdb/sdf/`：成功重建的分子结构文件。

### 5. 测试集采样与评估

如需使用 CrossDocked2020 测试集采样，请将数据整理到：

```text
${ONESCIENCE_DATASETS_DIR}/targetdiff/data/crossdocked_v1.1_rmsd1.0_pocket10
${ONESCIENCE_DATASETS_DIR}/targetdiff/data/crossdocked_pocket10_pose_split.pt
```

单条测试集样本采样：

```bash
python -m scripts.sample_diffusion configs/sampling.yml \
    -i 0 \
    --batch_size 50 \
    --result_path outputs
```

批量采样：

```bash
bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 0 0
```

生成结果评估：

```bash
python scripts/evaluate_diffusion.py outputs \
    --docking_mode none \
    --protein_root "${ONESCIENCE_DATASETS_DIR}/targetdiff/data/test_set"
```

若使用 `vina_score`、`vina_dock` 或 `qvina`，请提前安装相应 docking 工具并确认命令可用。

### 6. 模型训练

扩散模型训练需要 CrossDocked2020 数据：

```bash
python -m scripts.train_diffusion configs/training.yml \
    --device cuda \
    --logdir logs_diffusion \
    --tag targetdiff_train
```

亲和力预测模型训练需要 PDBbind 数据。完成口袋提取和数据划分后，可运行：

```bash
python -m scripts.property_prediction.train_prop configs/prop/pdbbind_general_egnn.yml \
    --device cuda \
    --logdir logs_prop \
    --tag targetdiff_prop_train \
    --dataset.path data/pdbbind_v2020_processed/pocket_10_refined \
    --dataset.split data/pdbbind_v2020_processed/pocket_10_refined/split.pt
```

PDBbind 口袋提取与划分脚本位于：

- `scripts/property_prediction/extract_pockets.py`
- `scripts/property_prediction/pdbbind_split.py`

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
