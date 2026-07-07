# EagleMeshTransformer

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 EagleMeshTransformer 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/EagleMeshTransformer" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

EagleMeshTransformer 是 OneScience 中面向 EAGLE 大规模二维非定常流体数据集的网格 Transformer 标准运行包。模型实现基于 GraphViT，使用节点聚类、图池化和全局注意力来预测随时间演化的速度场和压力场。

本仓库包含训练、评估、可视化入口、适配后的配置、split 文件、预检脚本和机器可读 Manifest。运行时必须配套 ModelScope 数据集 `OneScience/eagle`，并把 `ONESCIENCE_EAGLE_DATA_DIR` 指向该数据集仓库中的 `data/Eagle_dataset`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, turbulent_flow, unsteady_flow, mesh_transformer |
| 任务 | eagle_flow_forecasting |
| 任务标签 | train, inference, evaluation, visualization |
| 主平台资源 | https://modelscope.cn/models/OneScience/EagleMeshTransformer |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/EagleMeshTransformer` |
| 必需模型文件 | `train_graphvit.py`, `eval_graphvit.py`, `conf/graphvit_eagle.yaml`, `scripts/preflight_check.py`, `splits/*.txt` |
| 必需数据集 | `OneScience/eagle` |
| 支持能力 | 预检、训练、分布式训练、评估、推理式评测、可视化 |
| 最小验证 | `ONESCIENCE_EAGLE_DATA_DIR=/path/to/OneScience_eagle/data/Eagle_dataset python scripts/preflight_check.py` |

能力和命令要求

| 能力 | 必须提供 |
|---|---|
| `inference` | `python eval_graphvit.py`，需要测试数据和 `checkpoints/eagle_graphvit/best_model.pth` |
| `train` | `python train_graphvit.py`，使用 `conf/graphvit_eagle.yaml` 和 `OneScience/eagle` |
| `finetune` | 暂未声明 |
| `evaluate` | `python eval_graphvit.py` 输出 N-RMSE 日志 |
| `visualize` | `python eval_graphvit.py` 输出 `animation_results/comparison_*.gif` |
| `deploy` | 暂未声明 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 文档 | 人类用户和大模型入口，说明文件、关系、下载、预检和运行方式 | 是 | 全部能力 | 模型包根目录 | 中文正文 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明模型文件、数据集关系、命令、输出和诊断 | 是 | 全部能力 | 模型包根目录 | 大模型必须解析 |
| `conf/graphvit_eagle.yaml` | 配置文件 | GraphViT/EagleMeshTransformer 的数据、模型和训练配置 | 是 | 预检、训练、评估 | `conf/graphvit_eagle.yaml` | 已将数据目录适配为 `${ONESCIENCE_EAGLE_DATA_DIR}` |
| `scripts/preflight_check.py` | 预检脚本 | 检查配置、数据路径、split、NPZ schema、三角网格和聚类文件 | 是 | 预检 | `scripts/preflight_check.py` | 不启动训练 |
| `train_graphvit.py` | 训练脚本 | 构建 EagleDatapipe 和 GraphViT，训练并保存 checkpoint | 是 | 训练 | `train_graphvit.py` | 输出到 `checkpoints/eagle_graphvit/` |
| `eval_graphvit.py` | 评估脚本 | 加载 checkpoint，计算 N-RMSE 并生成 GIF | 是 | 推理、评估、可视化 | `eval_graphvit.py` | 需要 `best_model.pth` |
| `clusterize_eagle.py` | 工具脚本 | 重新生成聚类文件的参考脚本 | 否 | 数据准备 | `clusterize_eagle.py` | 默认路径需按本地数据位置调整 |
| `splits/train.txt` | Split 文件 | 训练样本相对路径列表 | 是 | 训练 | `splits/train.txt` | 947 条 |
| `splits/valid.txt` | Split 文件 | 验证样本相对路径列表 | 是 | 训练、验证 | `splits/valid.txt` | 118 条 |
| `splits/test.txt` | Split 文件 | 测试样本相对路径列表 | 是 | 评估、可视化 | `splits/test.txt` | 118 条 |
| `slurm.sh` | 集群脚本 | Slurm 训练示例 | 否 | 分布式训练 | `slurm.sh` | 需按集群调整 |

## Manifest

机器可读 Manifest 位于仓库根目录 `onescience_run_manifest.yaml`。修改模型入口、配置路径、下载命令、数据集 ID、运行矩阵或输出路径后，必须同步更新该文件，并建议执行 YAML 解析和 `command_refs` 校验。

## 模型 vs 数据集关系

本模型必须配套数据集 `OneScience/eagle` 使用。模型 Manifest 的 `relations.required_datasets` 已声明完整 `resource_ref`，指向 `https://modelscope.cn/datasets/OneScience/eagle`、`README.md` 和 `onescience_run_manifest.yaml`。数据集 Manifest 也通过 `relations.compatible_models` 反向声明适配模型 `OneScience/EagleMeshTransformer`。

模型默认读取 `Cre/Spl/Tri/<case>/<traj>/sim.npz`、`triangles.npy` 和 `constrained_kmeans_40.npy`。当前整理后的配置与本次数据完全匹配：数据集中包含 1200 个样本目录、7200 个数据文件，训练/验证/测试 split 文件均可在数据目录中解析。

## 文件与下载

下载模型：

```bash
modelscope download --model OneScience/EagleMeshTransformer
```

下载数据集：

```bash
modelscope download --dataset OneScience/eagle
```

如果使用 `--cache_dir` 下载，请先 `cd` 到实际下载后的模型包根目录再执行运行命令。数据集下载后，将环境变量指向数据集仓库中的 `data/Eagle_dataset` 目录：

```bash
export ONESCIENCE_EAGLE_DATA_DIR=/path/to/OneScience_eagle/data/Eagle_dataset
```

## 环境安装

```bash
bash install.sh cfd
```

还需要运行环境中可导入 `onescience`、`torch`、`numpy`、`matplotlib`、`pillow`、`tqdm` 和 `ruamel.yaml`。

## 运行流程

### 1. 环境预检

```bash
python - <<'PY'
import torch, numpy, matplotlib
import onescience
print("environment ok")
PY
```

### 2. 下载

```bash
modelscope download --model OneScience/EagleMeshTransformer
modelscope download --dataset OneScience/eagle
```

### 3. 应用运行包和准备文件

```bash
cd /path/to/downloaded/OneScience_EagleMeshTransformer
export ONESCIENCE_EAGLE_DATA_DIR=/path/to/downloaded/OneScience_eagle/data/Eagle_dataset
```

### 4. 运行前预检

```bash
python scripts/preflight_check.py
```

成功时应看到 `[OK] model preflight completed`。

### 5. 运行

训练：

```bash
python train_graphvit.py
```

多卡训练：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=<num_GPUs> train_graphvit.py
```

评估、推理式测试和可视化：

```bash
python eval_graphvit.py
```

### 6. 验证输出

训练会在 `checkpoints/eagle_graphvit/best_model.pth` 保存 checkpoint。评估会输出 Final N-RMSE 日志，并在 `animation_results/` 下生成 `comparison_*.gif`。

## 预检与诊断

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `ONESCIENCE_EAGLE_DATA_DIR is not set` | 未设置数据集目录环境变量 | 设置为 `OneScience/eagle` 数据集仓库的 `data/Eagle_dataset` 目录 |
| `Data path not found` | 环境变量指向错误或数据未下载完整 | 确认存在 `data/Eagle_dataset/Cre`, `data/Eagle_dataset/Spl`, `data/Eagle_dataset/Tri` |
| `Cluster file not found` | 缺少 `constrained_kmeans_40.npy` 或配置与数据不一致 | 运行数据集验证脚本，或同步修改 `n_cluster` 并更新 Manifest |
| `Checkpoint file not found` | 尚未训练或未放入 checkpoint | 先执行 `python train_graphvit.py`，或放入兼容的 `best_model.pth` |
| `ModuleNotFoundError` | OneScience CFD 环境或依赖缺失 | 安装 OneScience CFD 运行环境和 Python 依赖 |

## 输出说明

| 输出路径 | 说明 | 成功标准 |
|---|---|---|
| `checkpoints/eagle_graphvit/best_model.pth` | 训练产生的最优模型权重 | 文件存在且可由 `eval_graphvit.py` 加载 |
| `animation_results/comparison_*.gif` | 评估阶段生成的真值与预测对比动画 | `python eval_graphvit.py` 完成后生成 |
| 控制台 Final N-RMSE | 压力和速度归一化误差日志 | 日志中出现 `Final N-RMSE` |

## 限制与适用范围

本标准包默认使用 EAGLE v1 数据结构、`n_cluster=40` 和 990 个时间步样本。评估脚本需要已有 `checkpoints/eagle_graphvit/best_model.pth`；本模型仓库当前不内置预训练 checkpoint。若切换聚类数量、split 或数据版本，必须同步修改 `conf/graphvit_eagle.yaml` 和 `onescience_run_manifest.yaml`。

## 引用与许可证

参考论文：`EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers`，OpenReview `https://openreview.net/forum?id=ZZTkLDRmkg`，arXiv `https://arxiv.org/abs/2302.10803`。当前整理包未在原始目录中发现明确许可证文件，公开分发前应补充许可证信息。
