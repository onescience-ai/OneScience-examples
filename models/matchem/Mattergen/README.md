---
license: mit
tasks:
  - crystal-structure-generation
  - materials-generation
  - fine-tuning
frameworks:
  - pytorch
language:
  - en
  - zh
tags:
  - OneScience
  - MatterGen
  - 材料科学
  - 晶体生成
  - 扩散模型
  - 图神经网络
  - 条件生成
  - 训练
  - 微调
---

<p align="center">
  <strong>
    <span style="font-size: 30px;">MatterGen</span>
  </strong>
</p>

# 模型介绍

MatterGen 是微软研究院提出的无机材料生成模型，可联合生成元素组成、晶胞和周期性原子坐标，并支持根据材料属性生成候选晶体结构。

论文：*MatterGen: a generative model for inorganic materials design*<br>
参考实现：https://github.com/microsoft/mattergen

# 模型描述

MatterGen 使用扩散模型学习晶体的原子类型、分数坐标和晶格分布，可进行无条件晶体生成，也可根据化学体系、空间群、磁密度、带隙和体积模量等属性进行条件生成。

本仓库包含 OneScience 适配后的 MatterGen 模型、扩散与采样代码，以及训练、属性微调、晶体生成和数据转换入口。模型代码位于 `model/`，数据处理、通用网络层和属性 embedding 由 OneScience MatChem 提供。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 无条件晶体生成 | 使用基础 checkpoint 生成新的候选晶体结构 |
| 属性条件生成 | 根据磁密度、带隙、体积模量等目标属性生成结构 |
| 固定组成结构预测 | 给定目标组成后搜索可能的晶体结构 |
| 从头训练 | 使用 MP-20 或兼容 MatterGen cache 的数据训练模型 |
| 属性微调 | 在预训练模型上增加属性 adapter 并进行微调 |
| 环境连通性验证 | 通过单样本生成检查 OneScience、checkpoint 和 GPU/DCU 可用性 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于导入和配置连通性验证，但完整生成和训练速度较慢。
- DCU 需要加载与 PyTorch 构建版本匹配的 DTK。本示例已在 DTK 25.04.2 和 `torch 2.5.1+das.opt1.dtk25042` 环境完成单样本生成验证。

### 下载模型包

```bash
modelscope download --model OneScience/Mattergen --local_dir ./Mattergen
cd Mattergen
```

也可使用 ModelScope SDK：

```python
from modelscope import snapshot_download

model_dir = snapshot_download("OneScience/Mattergen")
print(model_dir)
```

### 安装运行环境

**DCU 环境**

```bash
# 请首先激活 DTK 及 conda
module load compiler/dtk/25.04.2
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持 uv 安装
pip install onescience[matchem-dcu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
# 请首先激活 conda
conda create -n onescience311 python=3.11 -y \
  libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持 uv 安装
pip install onescience[matchem-gpu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

MatterGen 训练使用转换后的数据 cache。MP-20 数据集已发布至 ModelScope，可直接下载：

```bash
modelscope download --dataset OneScience/mp20 --local_dir ./datasets/mp20
```

下载后的 MatterGen cache 位于：

```text
datasets/mp20/data/MP20/cache/mp_20/
├── train/
├── val/
└── test/
```

下载后不能直接使用默认训练 YAML，需要在 YAML 的 `args` 下添加实际数据路径。例如，修改 `demo/configs/train_8dcu.yaml`：

```yaml
args:
  data_module: mp_20
  data_module.root_dir: ./datasets/mp20/data/MP20/cache/mp_20
```

进行属性微调时，同样在 `demo/configs/finetune_dft_mag_density_smoke.yaml` 的 `args` 下添加 `data_module.root_dir`。如果数据下载到了其他位置，请将其改为对应的绝对路径。


自定义 CSV 数据需要先转换：

```bash
python csv_to_dataset.py \
  --csv-folder /path/to/csv_folder \
  --dataset-name my_dataset \
  --cache-folder ./datasets/cache
```

### 训练权重

通过前面的 ModelScope 命令下载模型包后，预训练 checkpoint 位于 `weight/`：

```text
weight/
├── mattergen_base/
│   ├── config.yaml
│   └── checkpoints/
│       └── last.ckpt
└── dft_mag_density/
    ├── config.yaml
    └── checkpoints/
        └── last.ckpt
```

仓库还提供 `chemical_system`、`dft_band_gap`、`dft_mag_density`、`ml_bulk_modulus`、`space_group` 等属性条件生成 checkpoint。使用 Demo YAML 时，请将 `demo/configs/generate_base.yaml` 或 `demo/configs/generate_dft_mag_density.yaml` 中的 `checkpoint` 修改为对应的 `./weight/<模型名>` 路径；微调配置则修改 `demo/configs/finetune_dft_mag_density_smoke.yaml` 中的 `adapter.model_path`。

### 推理

无条件生成一个晶体结构：

```bash
python generate.py \
  --checkpoint ./weight/mattergen_base \
  --output outputs/generate/mattergen_base \
  --batch-size 1 \
  --num-batches 1
```

生成结果包括：

```text
outputs/generate/mattergen_base/
├── generated_crystals.extxyz
└── generated_crystals_cif.zip
```

磁密度条件生成需要使用匹配的属性 checkpoint：

```bash
python generate.py \
  --checkpoint ./weight/dft_mag_density \
  --output outputs/generate/dft_mag_density \
  --batch-size 1 \
  --num-batches 1 \
  --properties '{"dft_mag_density": 0.15}'
```

也可以使用 Demo YAML：

```bash
cd demo
bash run.sh --config configs/generate_base.yaml
bash run.sh --config configs/generate_dft_mag_density.yaml
```

### 训练

使用 MP-20 cache 提交 8 卡 DCU 训练：

```bash
cd demo
bash run.sh --config configs/train_8dcu.yaml --submit
```

也可从仓库根目录直接使用 Hydra 参数启动：

```bash
python train.py \
  data_module=mp_20 \
  data_module.root_dir=./datasets/mp20/data/MP20/cache/mp_20 \
  trainer.devices=1 \
  data_module.batch_size.train=4
```

### 微调

仓库提供磁密度属性微调冒烟配置，默认只运行一个训练 batch 和一个验证 batch：

```bash
cd demo
bash run.sh --config configs/finetune_dft_mag_density_smoke.yaml
```

正式微调时，复制该 YAML 并修改 `adapter.model_path`、`data_module.properties`、属性 embedding、batch size 和训练轮数，同时移除 `trainer.limit_train_batches` 与 `trainer.limit_val_batches`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

---

# 引用与许可证

- MatterGen 相关代码来自 OneScience 项目中的 MatChem 示例实现，并参考了上游 MatterGen 项目（https://github.com/microsoft/mattergen）。上游 MatterGen 代码以 [MIT License](https://github.com/microsoft/mattergen/blob/main/LICENSE) 发布。
- 如果在科研工作中使用 MatterGen 训练或生成结果，建议引用 MatterGen 原始论文、OneScience 相关项目信息和实际使用的数据集来源。
