<p align="center">
  <strong>
    <span style="font-size: 30px;">STATE</span>
  </strong>
</p>

# 模型介绍

STATE 是 Arc Institute 提出的单细胞虚拟细胞模型体系，用于学习细胞状态表示并预测细胞在基因、药物或细胞因子扰动后的响应。

原始论文：Predicting cellular responses to perturbation across diverse contexts with State  
https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2

# 模型描述

STATE 包含两条可以独立使用、也可以串联的模型主线：
- **State Embedding（SE）**：把单细胞基因表达数据编码为低维细胞表征。
- **State Transition（ST）**：根据对照细胞、扰动条件、细胞类型和批次等协变量，预测扰动后的细胞状态或基因表达。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| SE 细胞表征 | 使用 SE-600M 将任意兼容 h5ad 转换为 `X_state` |
| SE 训练验证 | 使用 SE-167M-Human-smoke 构建 profile 并完成短训练 |
| CRISPR 基因扰动预测 | 使用 Replogle 数据和 ST-HVG/ST-SE 权重预测基因敲低响应 |
| 药物扰动预测 | 使用 Tahoe 数据和 Tahoe 权重预测药物×剂量响应 |
| 免疫刺激预测 | 使用 Parse PBMC 数据和 Parse 权重预测 cytokine 响应 |
| few-shot 评估 | 在同一细胞类型中留出部分扰动进行测试 |
| zero-shot 评估 | 留出完整细胞类型、细胞系或 donor 进行测试 |
| 新数据推理 | 使用 `infer_transition.py` 对不在训练 TOML 中的新 h5ad 推理 |
| 配置数据预测 | 使用 `predict_transition.py` 在 TOML 定义的数据划分上预测并运行 cell-eval |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 NVIDIA GPU 或海光 DCU 运行模型训练和推理。
- CPU 可用于脚本语法、导入、配置和小数据连通性验证。
- SE-600M checkpoint 约 11.5 GB，加载和推理需要足够的主存与显存。
- Parse 全量数据约 343 GB，需要额外考虑读取内存、缓存和输出空间。
- DCU 用户需要预先安装与集群匹配的 DTK 和 OneScience 推荐环境。

**软件要求**

- Python 3.10 以上；如需与原 `arc-state 0.11.2` 完全一致，建议使用 Python 3.11。
- OneScience 生信依赖，包括 PyTorch、Lightning、AnnData、Scanpy、cell-load 和 cell-eval。
- LanceDB 查询为可选功能，需要额外安装 `lancedb`。

安装 OneScience 生信环境：

```bash
conda create -n onescience-state python=3.11 -y
conda activate onescience-state
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

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

### 3.1 下载数据集和设置资产路径-数据集已上传魔搭平台

```bash
cd ./state

modelscope download --dataset OneScience/State_datasets
export ONESCIENCE_DATASETS_DIR=path/to/State_datasets

```
### 训练权重

训练权重已包含在'./weights'文件夹内，包含SE-600M embedding权重，ST&SE--PARSE&THAOE&Replogle系列权重。

### 3.3 主线一：SE-600M + SE smoke

该主线验证 SE profile、SE 训练入口和预训练 SE-600M transform。

#### 步骤一：生成 train/val manifest 并构建 profile

```bash
bash scripts/se_smoke_preprocess.sh
```
默认输入：

```text
${ONESCIENCE_DATASETS_DIR}/State_dataset/SE-167M-Human-smoke/
```

默认输出：

```text
${STATE_OUTPUT_DIR}/se_smoke/manifests/
${STATE_OUTPUT_DIR}/se_smoke/profile/
${STATE_OUTPUT_DIR}/se_smoke/config.yaml
```

#### 步骤二：训练 SE smoke 模型

```bash
bash scripts/se_smoke_train.sh \
  experiment.val_check_interval=1 \
  experiment.limit_val_batches=1 \
  model.batch_size=2
```

这一步用于验证训练接口，不用于复现官方 SE-600M。

#### 步骤三：使用 SE-600M 生成 `X_state`

```bash
bash scripts/se600m_smoke_transform.sh
```

默认输入和输出：

```text
输入：${ONESCIENCE_DATASETS_DIR}/State_dataset/SE-167M-Human-smoke/19k_human_filtered_scbasecount/SRX10188960.h5ad
输出：${STATE_OUTPUT_DIR}/se600m_smoke/SRX10188960_x_state.h5ad
```

替换输入输出：

```bash
export STATE_SE_INPUT=/path/to/input.h5ad
export STATE_SE_OUTPUT=/path/to/output_x_state.h5ad
bash scripts/se600m_smoke_transform.sh --batch-size 8
```

如需替换 SE 权重，直接使用通用 runner：

```bash
python scripts/runner/transform_embedding.py \
  --checkpoint /path/to/se_checkpoint.ckpt \
  --protein-embeddings /path/to/protein_embeddings.pt \
  --input /path/to/input.h5ad \
  --output /path/to/output_x_state.h5ad \
  --embed-key X_state
```

### 3.4 主线二：Replogle + ST-HVG

该主线使用 `X_hvg` 预测 CRISPR/基因扰动后的表达变化。

选择默认细胞系和划分：

```bash
export STATE_REPLOGLE_CELL_LINE=hepg2
export STATE_REPLOGLE_SPLIT_MODE=fewshot
```

可选细胞系：`hepg2`、`jurkat`、`k562`、`rpe1`。  
可选划分：`fewshot`、`zeroshot`。

#### 步骤一：生成 `X_hvg`

```bash
bash scripts/st_hvg_replogle_preprocess.sh
```

替换输入、输出和 HVG 数量：

```bash
export STATE_REPLOGLE_INPUT=/path/to/input.h5ad
export STATE_ST_HVG_INPUT=/path/to/input_x_hvg.h5ad
export STATE_NUM_HVGS=2000
bash scripts/st_hvg_replogle_preprocess.sh
```

#### 步骤二：短训练

```bash
bash scripts/st_hvg_replogle_train.sh \
  training.max_steps=10 \
  training.batch_size=2 \
  training.val_freq=5
```

#### 步骤三：使用下载权重预测并评估

```bash
bash scripts/st_hvg_replogle_predict.sh --profile minimal
```

#### 步骤四：推理

```bash
bash scripts/st_hvg_replogle_infer.sh
```

### 3.5 主线三：Replogle + SE-600M + ST-SE

该主线先用 SE-600M 生成 `X_state`，再使用 ST-SE-Replogle 预测基因扰动响应。

#### 步骤一：为 Replogle 生成 `X_state`

```bash
bash scripts/st_se_replogle_preprocess.sh
```

#### 步骤二：可选 ST-SE 短训练

```bash
bash scripts/st_se_replogle_prepare_all.sh

STATE_ST_SE_DATA_DIR=${STATE_OUTPUT_DIR}/st_se_replogle/data_all \
bash scripts/st_se_replogle_train.sh \
  training.max_steps=10 \
  training.batch_size=2 \
  training.val_freq=5
```

#### 步骤三：使用 ST-SE-Replogle 预测并评估

如果使用官方权重需要注意数据匹配，以下为上述流程生成的数据进行预测评估的示例；

```bash
python scripts/runner/predict_transition.py \
  --output-dir ${STATE_OUTPUT_DIR}/st_se_replogle/runs/st_se_replogle \
  --checkpoint last.ckpt \
  --toml ${STATE_OUTPUT_DIR}/st_se_replogle/configs/train.toml \
  --profile minimal
```

#### 步骤四：完整联合推理

```bash
bash scripts/st_se_replogle_infer.sh
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- STATE 原始论文：Predicting cellular responses to perturbation across diverse contexts with State。
- 论文地址：https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2
- 原始项目：https://github.com/ArcInstitute/state
- 原始 STATE 源码采用 CC BY-NC-SA 4.0 许可。
- STATE 模型权重和输出受 Arc Research Institute State Model Non-Commercial License 和 Acceptable Use Policy 约束。
- Parse 数据的部分用途可能需要 Parse Biosciences 许可，使用前应同时检查数据目录中的许可证文件。
- 在科研工作中使用本模型、权重或数据时，应引用 STATE 原始论文、对应数据集来源和 OneScience 项目信息。



