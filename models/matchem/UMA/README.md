<p align="center">
  <strong>
    <span style="font-size: 30px;">UMA</span>
  </strong>
</p>

# 模型介绍

UMA（Universal Materials Interaction Model）是面向材料与催化体系的通用机器学习原子间势模型，基于等变图神经网络构建，可对原子结构进行能量和受力预测。


# 模型描述

UMA 基于等变图神经网络架构，使用 OC20、OC22、OC25、OMat、OMOL、ODAC、OMC 等多种材料与催化数据集进行训练，面向催化吸附、无机材料、分子体系和 MOFs 等场景开展能量与受力预测及结构优化。

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| OC20 能量和力微调 | 使用标准配置读取 OC20 微调数据并训练 UMA 模型 |
| OC22 氧化物催化微调 | 使用标准配置读取 OC22 微调数据并训练 UMA 模型（仅限 1P2） |
| OC25 （电）催化微调 | 使用标准配置读取 OC25 微调数据并训练 UMA 模型（仅 1P2） |
| OMat 无机材料微调 | 使用标准配置读取 OMat 微调数据并训练 UMA 模型 |
| OMOL 分子+聚合物微调 | 使用标准配置读取 OMOL 微调数据并训练 UMA 模型 |
| ODAC MOFs 微调 | 使用标准配置读取 ODAC 微调数据并训练 UMA 模型 |
| OMC 分子晶体微调 | 使用标准配置读取 OMC 微调数据并训练 UMA 模型 |
| 训练流程预检 | 检查配置、数据路径、运行脚本和 checkpoint 放置位置 |
| 催化吸附体系建模 | 参考 OC20/OC22/OC25 任务进行吸附/催化表面体系训练与推理 |
| 推理脚本参考 | 使用上游推理示例进行晶体弛豫、吸附体系弛豫或分子 MD 改造 |
| 自有数据迁移 | 将 ASE 可读结构转换为 UMA 微调数据后替换训练和验证路径 |



# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行微调训练。
- CPU 可以用于配置和数据路径预检，不建议用于正式 UMA 训练。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/UMA --local_dir ./UMA
cd UMA
```

### 安装运行环境

**DCU环境**
```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```


### 训练数据集介绍

本仓库不内置训练数据。以下以 **OC20 微调**为例说明流程，OC22、OC25、OMat、OMOL、ODAC、OMC 等其他任务流程相同，只需替换 `--uma-task` 和数据路径即可。


```bash
modelscope download --dataset OneScience/oc20 --local_dir ./data
```


**数据格式转换**

下载的原始数据通常是 `.extxyz` 文件，需要先用 `scripts/create_uma_finetune_dataset.py` 转换为 ASE-lmdb 格式，并计算 `elem_refs` 和 `normalizer_rmsd`。该脚本支持以下任务：

| 任务 | 说明 |
| --- | --- |
| `oc20` | 催化（示例） |
| `oc22` | 氧化物催化（仅限 1P2） |
| `oc25` | （电）催化（仅 1P2） |
| `omat` | 无机材料 |
| `omol` | 分子 + 聚合物 |
| `odac` | MOFs |
| `omc` | 分子晶体 |


```bash
python scripts/create_uma_finetune_dataset.py \
    --train-dir data/oc20/s2ef_200k_uncompressed \
    --val-dir data/oc20/s2ef_val_id_uncompressed \
    --uma-task oc20 \
    --regression-tasks ef \
    --output-dir data/oc20_finetune \
    --num-workers 8
```

转换后生成：

```text
data/oc20_finetune/
├── train/                 # ASE-lmdb 训练数据
├── val/                   # ASE-lmdb 验证数据
└── data/                  # 生成的数据配置 yaml
    └── uma_conserving_data_task_energy_force.yaml
```

然后用 `scripts/update_demo_config.py` 把生成的 `elem_refs`、`normalizer_rmsd` 和数据路径更新到 demo 配置文件：

```bash
python scripts/update_demo_config.py --demo-config demo/configs/oc20_ef_4dcu.yaml
```

`demo/run.sh` 会自动将仓库根目录作为 `ONESCIENCE_DATASETS_DIR`，因此配置文件中的相对路径会自动匹配。

### 训练权重

本仓库已包含旋转基文件 `weight/Jd.pt`。UMA 预训练 checkpoint（如 `uma-s-1p1_converted.pt`）需从 fairchem 官方仓库下载并按 UMA 格式转换后放到（即将上传）：

```text
weight/uma-s-1p1_converted.pt
```

- fairchem 官方仓库：https://github.com/facebookresearch/fairchem

`demo/run.sh` 和 `inference/` 下的示例脚本都会自动检测 `weight/Jd.pt` 并设置 `ONESCIENCE_UMA_JD_PATH`。


### 微调

```bash
bash demo/run.sh --config demo/configs/oc20_ef_4dcu.yaml
```

### 推理
```bash
python inference/run_molecular_md.py
```
需自行指定预训练权重的路径。

## OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

---

## 引用与许可证

- UMA 示例代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 fairchem 项目（https://github.com/facebookresearch/fairchem）。上游 fairchem 仓库软件以 [MIT License](https://fair-chem.github.io/core/install.html#license) 发布；fairchem 各模型 checkpoint 和数据集可能带有各自独立的许可证，使用时请遵循对应说明。
- 如果在科研工作中使用 UMA 微调或推理结果，建议引用 UMA/相关通用材料相互作用模型方法、fairchem/OneScience 相关项目信息和实际使用的数据集来源。

