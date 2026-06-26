# LagrangianMGN

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 LagrangianMGN 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/LagrangianMGN" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

LagrangianMGN 是面向拉格朗日粒子动力学的 MeshGraphNet 标准运行包。它把粒子位置历史、粒子类型和边界距离编码成图神经网络输入，预测下一步加速度，并通过自回归 rollout 生成后续粒子轨迹。

本仓库整理自 `onescience/examples/cfd/Lagrangian_MGN/`，已按 OneScience ModelScope 运行标准收敛到 DeepMind Lagrangian 数据中的 2D Water 子集。配套数据集仓库是 `OneScience/lagrangian`，其中包含 `data/Water/{train,valid,test}.tfrecord` 和 `metadata.json`。

该包支持运行前预检、Water 数据训练、基于训练 checkpoint 的推理、误差评估和 GIF 可视化。仓库本身不上传预训练 checkpoint，推理前需要先训练或提供 `resume_dir` 指向已有 checkpoint。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, particle_simulation, lagrangian_dynamics, graph_neural_network |
| 任务 | lagrangian_particle_simulation |
| 任务标签 | train, inference, evaluation, visualization |
| 主平台资源 | https://modelscope.cn/models/OneScience/LagrangianMGN |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/Lagrangian_MGN` |
| 必需模型文件 | `train.py`, `inference.py`, `conf/`, `scripts/preflight_lagrangian_mgn.py` |
| 必需数据集 | `OneScience/lagrangian` |
| 支持能力 | 预检、训练、推理、评测、可视化 |
| 最小验证 | `python scripts/preflight_lagrangian_mgn.py --repo-root . --data-dir "${ONESCIENCE_LAGRANGIAN_DATA_DIR:-data/Water}"` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、数据集关系、命令和输出 | 是 | 全部能力 | 模型包根目录 | 修改运行包时必须同步更新 |
| `onescience_relations.yaml` | 关系索引 | 声明 `OneScience/lagrangian` 是必需数据集 | 是 | 自动索引 | 模型包根目录 | 与 Manifest relations 保持一致 |
| `train.py` | 运行脚本 | Water 数据训练入口 | 是 | 训练 | 模型包根目录 | Hydra 配置入口 |
| `inference.py` | 运行脚本 | 使用 checkpoint 执行 rollout、评测和 GIF 可视化 | 是 | 推理、评测、可视化 | 模型包根目录 | 需要 `resume_dir` 有 checkpoint |
| `loggers.py` | 运行脚本 | 训练和推理日志工具 | 是 | 训练、推理 | 模型包根目录 | 由入口脚本导入 |
| `conf/config.yaml` | 配置文件 | 标准化默认配置，默认读取 Water 数据 | 是 | 预检、训练、推理 | `conf/config.yaml` | 已适配 `ONESCIENCE_LAGRANGIAN_DATA_DIR` |
| `conf/experiment/water.yaml` | 配置文件 | 2D Water 实验配置 | 是 | 训练、推理 | `conf/experiment/water.yaml` | 当前标准包只声明 Water |
| `scripts/preflight_lagrangian_mgn.py` | 预检脚本 | 检查配置、数据文件、metadata schema、归一化统计和 TFRecord 可读性 | 是 | 预检 | `scripts/preflight_lagrangian_mgn.py` | TensorFlow 可用时会读取首条记录 |
| `slurm.sh` | 示例脚本 | Slurm 训练参考 | 否 | 训练 | 模型包根目录 | 可按集群修改 |

## Manifest

机器可读 Manifest 位于仓库根目录：

```text
onescience_run_manifest.yaml
```

大模型应先读取本 README 的“文件说明”和本节，然后解析 `onescience_run_manifest.yaml`。修改配置、文件路径、下载方式、命令、关系或输出时，必须同步更新 Manifest，并重新运行 YAML 解析和 command_refs 校验。

## 模型 vs 数据集关系

本模型必须配套数据集 `OneScience/lagrangian` 使用。模型 Manifest 的 `relations.required_datasets` 指向：

| 字段 | 值 |
|---|---|
| 数据集 ID | `OneScience/lagrangian` |
| 数据集 URL | https://modelscope.cn/datasets/OneScience/lagrangian |
| Manifest | `onescience_run_manifest.yaml` |
| 模型期望数据路径 | `data/Water` |
| 推荐环境变量 | `ONESCIENCE_LAGRANGIAN_DATA_DIR=<dataset_repo_root>/data/Water` |

数据集 Manifest 会通过 `relations.compatible_models` 反向声明适配模型 `OneScience/LagrangianMGN`。

## 文件与下载

下载模型包：

```bash
modelscope download --model OneScience/LagrangianMGN
```

下载数据集包：

```bash
modelscope download --dataset OneScience/lagrangian
```

如果使用 `modelscope download --cache_dir`，命令完成后请切换到实际下载后的模型包根目录再运行本仓库命令。数据集下载后推荐设置：

```bash
export ONESCIENCE_LAGRANGIAN_DATA_DIR=<dataset_repo_root>/data/Water
```

也可以把数据集仓库中的 `data/Water` 放到模型包根目录的 `data/Water`。

## 环境安装

OneScience 网站环境如果已经部署 CFD 依赖，可直接运行预检。环境缺失时使用 OneScience CFD 安装入口：

```bash
bash install.sh cfd
```

本示例读取 TFRecord，需要 TensorFlow；训练和推理需要 PyTorch、DGL、Hydra、OmegaConf、Matplotlib、Pillow、TQDM 等依赖。

## 运行流程

预检：

```bash
python scripts/preflight_lagrangian_mgn.py --repo-root . --data-dir "${ONESCIENCE_LAGRANGIAN_DATA_DIR:-data/Water}"
```

训练 Water：

```bash
python train.py +experiment=water data.data_dir="${ONESCIENCE_LAGRANGIAN_DATA_DIR:-data/Water}" resume_dir=./model/Water output=./outputs/water
```

推理和可视化：

```bash
python inference.py +experiment=water data.data_dir="${ONESCIENCE_LAGRANGIAN_DATA_DIR:-data/Water}" data.test.num_sequences=4 resume_dir=./model/Water output=./result/water/inference
```

推理前 `resume_dir` 必须包含训练得到的 checkpoint。推理脚本会在 `result/water/inference/animations` 下生成 GIF 和 `error.png`。

## 预检与诊断

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `Missing mandatory value: data.data_dir` | 没有使用标准化配置或未传入数据路径 | 设置 `ONESCIENCE_LAGRANGIAN_DATA_DIR`，或在命令中传入 `data.data_dir=...` |
| `TensorFlow is required for reading .tfrecord files` | 环境缺少 TensorFlow | 安装 OneScience CFD 依赖或 `pip install "tensorflow<=2.17.1"` |
| `checkpoint not found` | 推理前没有 checkpoint | 先运行训练命令，或把 `resume_dir` 指向已有 checkpoint |
| `Only batch size 1 is currently supported` | 推理 batch size 被改为非 1 | 保持 `test.batch_size=1` |
| 找不到 `metadata.json` 或 TFRecord | 数据集未下载或路径错误 | 确认 `OneScience/lagrangian` 已下载，并设置 `ONESCIENCE_LAGRANGIAN_DATA_DIR=<dataset_repo_root>/data/Water` |

## 输出说明

| 输出 | 路径 | 说明 |
|---|---|---|
| checkpoint | `model/Water` | 训练产生的模型、优化器和调度器状态 |
| Hydra 输出 | `outputs/water` | 训练日志和 Hydra 配置快照 |
| GIF 动画 | `result/water/inference/animations/*.gif` | 预测轨迹与真值轨迹对比 |
| 误差图 | `result/water/inference/animations/error.png` | rollout 序列位置误差 |

## 限制与适用范围

本标准包默认只覆盖 DeepMind Lagrangian 的 2D Water 子集。原始示例中的 Goop、Sand、WaterRamps、MultiMaterial 和 Water 3D 配置未作为本次自动运行场景上传，因为当前数据集仓库不包含对应数据。仓库不包含预训练 checkpoint，推理需要先训练或外部提供 checkpoint。

## 引用与许可证

请引用原始 Learning to Simulate / MeshGraphNet 相关工作以及 OneScience 项目。当前整理包的许可证字段暂记为 unknown；使用数据和代码前请遵循原始数据集、论文和 OneScience 仓库的许可要求。
