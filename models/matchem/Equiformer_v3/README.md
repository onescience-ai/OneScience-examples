<p align="center">
  <strong>
    <span style="font-size: 30px;">Equiformer V3</span>
  </strong>
</p>

# 模型介绍

Equiformer V3 是面向三维原子体系的 SE(3) 等变图注意力势模型，可预测原子结构的能量、原子力和应力。

# 模型描述

Equiformer V3 基于等变图神经网络架构，在 Equiformer V2 的基础上改进了等变归一化、平滑截断注意力和 SwiGLU-S² 激活，面向材料势能面建模。当前发布包包含 OneScience 适配后的模型代码、预训练权重、单点推理和 OC20 训练示例。

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 单点能量、力和应力 | 使用 ASE 可读结构进行能量、力和应力预测 |
| 结构弛豫与形成能 | 使用模型和元素参考能进行结构优化及形成能计算 |
| 弹性张量 | 计算周期性材料的弹性性质 |
| 声子 | 计算周期性材料的声子性质 |
| OC20 S2EF 训练 | 使用 OneScience/oc20 的预处理数据训练能量和力 |
| 自有数据迁移 | 将自有数据转换为配置所需的 ASE-LMDB 后替换训练路径 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 NVIDIA GPU 或海光 DCU 运行推理和训练。
- CPU 可用于查看配置和准备数据，不建议用于正式训练。
- DCU 用户需要预先加载与当前 PyTorch 匹配的 DTK 环境。

### 下载模型包

```bash
modelscope download --model OneScience/Equiformer_v3 --local_dir ./Equiformer_v3
cd Equiformer_v3
```

### 安装运行环境

**DCU 环境**

```bash
# 请首先激活 DTK 及 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[matchem-dcu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
# 请首先激活 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[matchem-gpu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

### 训练数据集介绍

本模型仓库不内置训练数据。OC20 训练数据来自 ModelScope 数据集 **OneScience/oc20**，对应 Open Catalyst 2020（OC20）S2EF 任务。OC20 的原始数据、任务定义和引用信息见 [OC20 数据集论文](https://doi.org/10.1021/acscatal.0c04525) 和 [FAIR Chemistry OC20 文档](https://fair-chem.github.io/oc20/)。使用数据时请遵守 OneScience 数据集页面、Open Catalyst Project 及原始数据的许可证和使用条款。

```bash
modelscope download --dataset OneScience/oc20 --local_dir ./data
```

下载的数据需要按照 OneScience OC20 数据准备流程转换为训练配置使用的 ASE-LMDB 格式。当前配置读取的目录为：

```text
data/oc20_finetune/
├── train/                 # ASE-LMDB 训练数据
└── val/                   # ASE-LMDB 验证数据
```

如果下载的是 OC20 的 `.extxyz` 或其他 ASE 可读原始文件，可使用随包提供的 UMA 数据处理脚本进行转换：

```bash
python scripts/create_uma_finetune_dataset.py \
  --train-dir ./data/oc20/s2ef_200k_uncompressed \
  --val-dir ./data/oc20/s2ef_val_id_uncompressed \
  --uma-task oc20 \
  --regression-tasks ef \
  --output-dir ./data/oc20_finetune \
  --num-workers 8
```

脚本会生成 `train/`、`val/` ASE-LMDB 分片及 UMA 数据配置文件。输入文件必须包含 ASE calculator 提供的 energy 和 forces 标签；转换失败的文件会记录在每个输出分片对应的 `.failed` 日志中。`create_finetune_dataset.py` 是底层 ASE 数据转换实现，`create_uma_finetune_dataset.py` 是完整的 OC20/UMA 数据处理入口，两个脚本均来自 UMA 仓库并按原文件提供。

在 OneScience MatChem 数据根目录下，对应路径为：

```text
${ONESCIENCE_DATASETS_DIR}/matchem/oc20/uma_oc20_finetune/
├── train/
└── val/
```

如果数据位于其他路径，请修改 `demo/configs/oc20_scratch_8dcu.yaml` 和 `demo/configs/oc20_scratch_8dcu_smoke.yaml` 中的 `train`、`val` 字段，或设置 `ONESCIENCE_DATASETS_DIR`。Equiformer V3 的训练 YAML 仍需使用其自身的 `train`、`val` 和 `transforms.element_references.energy.file` 字段；UMA 脚本生成的 UMA 微调 YAML 不直接替代 Equiformer V3 训练配置。替换训练数据后，需要重新拟合 energy 元素参考系数：

```bash
python fit_element_references.py \
  --config demo/configs/oc20_scratch_8dcu.yaml \
  --output demo/reference_data/oc20_subset_energy_element_references.npz
```


### 训练权重

本仓库包含以下 Equiformer V3 权重：

```text
weight/
├── Jd.pt
├── mptrj_gradient.pt
├── omat24_direct.pt
├── omat24_gradient.pt
└── omat24-mptrj-salex_gradient.pt
```

| 权重 | 训练任务或数据域 | 用途 |
| --- | --- | --- |
| `mptrj_gradient.pt` | MPtrj gradient | MPtrj 材料结构推理或评估 |
| `omat24_direct.pt` | OMat24 direct + DeNS | OMat24 direct 模型推理或评估 |
| `omat24_gradient.pt` | OMat24 gradient | OMat24 gradient 模型推理或评估 |
| `omat24-mptrj-salex_gradient.pt` | OMat24 + MPtrj + sAlex gradient | 通用材料结构推理，推理脚本默认使用 |

`Jd.pt` 是 Wigner 旋转基文件，由入口脚本自动从 `weight/Jd.pt` 加载。

### 微调

运行 OC20 smoke 配置：

```bash
export ONESCIENCE_DATASETS_DIR=/path/to/onescience_datasets
bash demo/run.sh --config configs/oc20_scratch_8dcu_smoke.yaml
```

smoke 配置使用 8 个训练样本、8 个验证样本和一层缩小模型，仅执行一次更新，用于检查数据、分布式通信、前向、反向、优化器和 checkpoint 保存链路。

运行 OC20 完整配置：

```bash
bash demo/run.sh --config configs/oc20_scratch_8dcu.yaml
```

该配置使用 1 个节点、8 个 DCU 和 12 个 epoch。当前已验证的稳定路径为 FP32；控制实验中相同完整模型的 FP16/BF16 训练出现 DCU kernel VMFault，因此发布配置使用 `amp: false`。

### 推理

单点能量、力和应力：

```bash
python single_point.py --device cuda --output outputs/single_point.json
```

形成能：

```bash
python formation_energy.py --device cuda --output outputs/formation_energy.json
```

弹性张量：

```bash
python elastic.py --relax --device cuda --output outputs/elastic.json
```

声子：

```bash
python phonons.py \
  --supercell 3 3 3 \
  --bandpath GXWKGL \
  --device cuda \
  --output-dir outputs/phonons
```

自定义结构或权重：

```bash
python single_point.py --input structure.cif
python elastic.py --input POSCAR --output outputs/elastic.json
python phonons.py --input structure.cif --supercell 2 2 2
python single_point.py --checkpoint weight/omat24_gradient.pt
```

`--input` 支持 ASE 可读取的 CIF、POSCAR、XYZ 和 trajectory 等格式。四个推理脚本默认使用 `weight/omat24-mptrj-salex_gradient.pt`。

## OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

---

## 引用与许可证

- Equiformer V3 架构代码基于上游 MIT License，许可证见本仓库根目录 `LICENSE`。
- OC20 数据集采用其官方页面说明的 CC BY 4.0 许可证；数据集许可证不由本仓库的 MIT License 覆盖。
- 如果在科研工作中使用 Equiformer V3 或 OC20，请引用 Equiformer V3 论文、OC20 数据集论文及实际使用的数据集来源。

```bibtex
@article{equiformer_v3,
  title={EquiformerV3: Scaling Efficient, Expressive, and General SE(3)-Equivariant Graph Attention Transformers},
  author={Yi-Lun Liao and Alexander J. Hoffman and Sabrina C. Shen and Alexandre Duval and Sam Walton Norwood and Tess Smidt},
  journal={arXiv preprint arXiv:2604.09130},
  year={2026}
}

@article{oc20,
  title={Open Catalyst 2020 (OC20) Dataset and Community Challenges},
  author={Lowik Chanussot and Abhishek Das and Siddharth Goyal and others},
  journal={ACS Catalysis},
  year={2021},
  doi={10.1021/acscatal.0c04525}
}
```
