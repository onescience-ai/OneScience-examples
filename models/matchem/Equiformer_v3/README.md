<p align="center">
  <strong>
    <span style="font-size: 30px;">Equiformer V3</span>
  </strong>
</p>

# 模型介绍

Equiformer V3 是用于材料结构能量、原子力和应力预测的 SE(3) 等变图注意力势模型。OneScience 提供 ASE calculator 接口，可用于单点计算、形成能、弹性张量、声子、独立数据集评估以及 OC20 S2EF 训练。

上游实现：[atomicarchitects/equiformer_v3](https://github.com/atomicarchitects/equiformer_v3)

# 模型描述

Equiformer V3 基于等变图神经网络架构，面向三维原子结构学习材料势能面。不同预训练 checkpoint 对应不同材料数据域，使用时应选择与目标元素体系和标注设置接近的权重。本示例提供从头训练、checkpoint 初始化训练和恢复训练入口，以及单点、形成能、弹性和声子推理脚本。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 单点计算 | 预测周期结构的总能量、原子力和应力 |
| 形成能 | 使用元素参考能计算材料形成能 |
| 弹性张量 | 通过应变和模型应力拟合弹性性质 |
| 声子 | 使用 ASE 有限位移法计算声子性质 |
| OC20 S2EF 训练 | 使用预处理后的 OC20 ASE-LMDB 训练能量和原子力 |
| 分布式训练 | 支持单机多卡及 Slurm 任务入口 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行推理和训练。
- CPU 可用于导入和配置检查，不建议用于正式计算。
- DCU 用户需要加载与当前 PyTorch 匹配的 DTK 运行时。

### 获取运行资源

模型权重不放入社区示例仓库，请从 ModelScope 下载：

```bash
modelscope download --model OneScience/Equiformer_v3 --local_dir ./resources/Equiformer_v3
export ONESCIENCE_MODELS_DIR="$PWD/resources"
```

入口脚本默认读取：

```text
${ONESCIENCE_MODELS_DIR}/EquiformerV3/omat24-mptrj-salex_gradient.pt
```

如果下载目录名称或权重位置不同，请直接通过 `--checkpoint` 指定文件。

### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[matchem-dcu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[matchem-gpu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

### 训练数据集介绍

本示例不内置训练数据。OC20 训练数据来自 ModelScope 数据集 [OneScience/oc20](https://modelscope.cn/datasets/OneScience/oc20)，对应 Open Catalyst 2020 S2EF 任务。原始数据和引用信息见 [OC20 数据集论文](https://doi.org/10.1021/acscatal.0c04525)；使用时请遵守数据集许可证和使用条款。

```bash
modelscope download --dataset OneScience/oc20 --local_dir ./datasets/oc20
```

训练配置读取预处理后的 ASE-LMDB：

```text
${ONESCIENCE_DATASETS_DIR}/matchem/oc20/uma_oc20_finetune/
├── train/
└── val/
```

如果下载的是原始 ASE 可读文件，可使用随示例提供的 UMA 数据处理脚本：

```bash
python scripts/create_uma_finetune_dataset.py \
  --train-dir ./datasets/oc20/s2ef_200k_uncompressed \
  --val-dir ./datasets/oc20/s2ef_val_id_uncompressed \
  --uma-task oc20 \
  --regression-tasks ef \
  --output-dir ./datasets/oc20_finetune \
  --num-workers 8
```

其中 `scripts/create_finetune_dataset.py` 是底层 ASE 数据转换实现，两个脚本均来自 UMA 仓库。转换后请把 YAML 的 `train` 和 `val` 指向生成的 `train/`、`val/`，并使用 `fit_element_references.py` 重新拟合 energy 元素参考系数。

### 训练权重

社区仓库不包含权重。ModelScope 模型包提供：

| 权重 | 训练域 | 建议用途 |
| --- | --- | --- |
| `mptrj_gradient.pt` | MPtrj gradient | MPtrj 材料结构推理或评估 |
| `omat24_direct.pt` | OMat24 direct + DeNS | OMat24 direct 推理或评估 |
| `omat24_gradient.pt` | OMat24 gradient | OMat24 验证集评估 |
| `omat24-mptrj-salex_gradient.pt` | OMat24 + MPtrj + sAlex gradient | 通用材料推理，推理脚本默认使用 |

### 训练

先设置 OneScience 数据根目录：

```bash
export ONESCIENCE_DATASETS_DIR=/path/to/onescience_datasets
```

运行 OC20 smoke：

```bash
bash demo/run.sh --config configs/oc20_scratch_8dcu_smoke.yaml
```

smoke 使用 8 个训练样本、8 个验证样本和一层缩小模型，仅执行一次更新，用于检查数据、分布式通信、前向、反向、优化器和 checkpoint 保存链路。

运行完整 OC20 配置：

```bash
bash demo/run.sh --config configs/oc20_scratch_8dcu.yaml
```

该配置使用 1 个节点、8 个 DCU 和 12 个 epoch。当前稳定路径为 FP32；控制实验中的 FP16/BF16 训练出现 DCU kernel VMFault，因此 YAML 设置 `amp: false`。

### 微调

使用 `train.py` 的 `init_from_checkpoint` 模式，在 YAML 中指定 `initialization_checkpoint` 或 `checkpoint`，并将 `mode` 设置为 `init_from_checkpoint`。例如：

```yaml
mode: init_from_checkpoint
initialization_checkpoint: /path/to/checkpoint.pt
train: /path/to/train
val: /path/to/val
output: checkpoints/equiformer_v3_finetuned.pt
```

通过同一个入口运行：

```bash
python train.py --config path/to/finetune.yaml
```

### 推理

单点能量、力和应力：

```bash
python single_point.py --checkpoint "$ONESCIENCE_MODELS_DIR/EquiformerV3/omat24-mptrj-salex_gradient.pt"
```

形成能：

```bash
python formation_energy.py --checkpoint "$ONESCIENCE_MODELS_DIR/EquiformerV3/omat24-mptrj-salex_gradient.pt"
```

弹性张量：

```bash
python elastic.py --checkpoint "$ONESCIENCE_MODELS_DIR/EquiformerV3/omat24-mptrj-salex_gradient.pt" --relax
```

声子：

```bash
python phonons.py \
  --checkpoint "$ONESCIENCE_MODELS_DIR/EquiformerV3/omat24-mptrj-salex_gradient.pt" \
  --supercell 3 3 3 --bandpath GXWKGL
```

通过 `--input` 可读取 CIF、POSCAR、XYZ 等 ASE 支持的结构文件。`evaluate.py` 可用于 ASE DB 或 ASE-LMDB 独立数据集评估。

# 文件说明

| 路径 | 作用 |
| --- | --- |
| `model/` | Equiformer V3 模型包源码镜像 |
| `single_point.py` | 单点能量、力和应力推理 |
| `formation_energy.py` | 形成能计算 |
| `elastic.py` | 弹性张量计算 |
| `phonons.py` | 声子计算 |
| `evaluate.py` | 独立数据集评估 |
| `train.py` | 从头训练、checkpoint 初始化和恢复训练 |
| `finetune.py` | 兼容的 checkpoint 微调入口 |
| `scripts/` | UMA OC20 数据转换脚本 |
| `demo/run.sh` | YAML 驱动的本地和 Slurm 训练入口 |

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Equiformer V3 模型代码遵循上游项目许可证，详见 ModelScope 模型仓库。
- OC20 数据集遵循其官方数据集许可证，本示例仓库不重新分发数据。
- 使用 Equiformer V3 或 OC20 时，请引用对应模型论文和 OC20 数据集论文。
