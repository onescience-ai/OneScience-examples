# MeshGraphNet

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 MeshGraphNet 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/MeshGraphNet" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

MeshGraphNet 是面向二维圆柱绕流瞬态预测的图神经网络标准运行包。模型把非结构三角网格转换为 DGL 图，以节点速度、节点类型和边几何特征为输入，预测下一时刻速度差和压力，并可通过自回归推理生成涡街演化 GIF。

本仓库整理自 `onescience/examples/cfd/Vortex_shedding_mgn/`，已按 OneScience ModelScope 运行标准适配配套数据集 `OneScience/cylinder_flow`。仓库包含训练、推理、配置和预检脚本，不上传预训练 checkpoint；推理前需要先训练，或把兼容 checkpoint 放入 `checkpoints/`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, vortex_shedding, cylinder_flow, mesh_graph_network |
| 任务 | transient_cylinder_flow_surrogate |
| 任务标签 | train, inference, evaluation, visualization |
| 主平台资源 | https://modelscope.cn/models/OneScience/MeshGraphNet |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/Vortex_shedding_mgn` |
| 必需模型文件 | `train.py`, `inference.py`, `conf/mgn_cylinderflow.yaml`, `scripts/preflight_vortex_shedding_mgn.py` |
| 必需数据集 | `OneScience/cylinder_flow` |
| 支持能力 | 预检、训练、推理、评测、可视化 |
| 最小验证 | `python scripts/preflight_vortex_shedding_mgn.py --repo-root . --data-dir data/cylinder_flow --skip-tfrecord-read` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、数据集关系、命令和输出 | 是 | 全部能力 | 模型包根目录 | 修改运行包时必须同步更新 |
| `onescience_relations.yaml` | 关系索引 | 声明 `OneScience/cylinder_flow` 是必需数据集 | 是 | 自动索引 | 模型包根目录 | 与 Manifest relations 保持一致 |
| `train.py` | 运行脚本 | MeshGraphNet 训练入口 | 是 | 训练 | 模型包根目录 | 读取 `conf/mgn_cylinderflow.yaml` |
| `inference.py` | 运行脚本 | 使用 checkpoint 执行推理、评测和 GIF 可视化 | 是 | 推理、评测、可视化 | 模型包根目录 | 需要 `checkpoints/` 下有兼容 checkpoint |
| `conf/mgn_cylinderflow.yaml` | 配置文件 | 数据路径、样本数、模型结构和训练参数 | 是 | 预检、训练、推理 | `conf/mgn_cylinderflow.yaml` | 已适配默认读取 `data/cylinder_flow` |
| `scripts/preflight_vortex_shedding_mgn.py` | 预检脚本 | 检查配置、TFRecord、meta、stats 和首条记录可读性 | 是 | 预检 | `scripts/preflight_vortex_shedding_mgn.py` | 可加 `--skip-tfrecord-read` 做快速结构检查 |
| `slurm.sh` | 示例脚本 | Slurm 训练参考 | 否 | 训练 | 模型包根目录 | 可按集群修改 |

## Manifest

机器可读 Manifest 位于仓库根目录：

```text
onescience_run_manifest.yaml
```

大模型应先读取本 README 的“文件说明”和本节，然后解析 `onescience_run_manifest.yaml`。修改配置、文件路径、下载方式、命令、关系或输出时，必须同步更新 Manifest，并重新运行 YAML 解析和 command_refs 校验。

## 模型 vs 数据集关系

本模型必须配套数据集 `OneScience/cylinder_flow` 使用。模型 Manifest 的 `relations.required_datasets` 指向：

| 字段 | 值 |
|---|---|
| 数据集 ID | `OneScience/cylinder_flow` |
| 数据集 URL | https://modelscope.cn/datasets/OneScience/cylinder_flow |
| Manifest | `onescience_run_manifest.yaml` |
| 模型期望数据路径 | `data/cylinder_flow` |
| 可选环境变量 | `ONESCIENCE_CYLINDER_FLOW_DATA_DIR=<dataset_repo_root>/data/cylinder_flow` |

数据集 Manifest 会通过 `relations.compatible_models` 反向声明适配模型 `OneScience/MeshGraphNet`。

## 文件与下载

下载模型包：

```bash
modelscope download --model OneScience/MeshGraphNet
```

下载数据集包：

```bash
modelscope download --dataset OneScience/cylinder_flow
```

如果使用 `modelscope download --cache_dir`，命令完成后请切换到实际下载后的模型包根目录再运行本仓库命令。数据集下载后推荐把数据目录放到模型包根目录：

```bash
mkdir -p data
cp -a <dataset_repo_root>/data/cylinder_flow data/cylinder_flow
```

## 环境安装

OneScience 网站环境如果已经部署 CFD 依赖，可直接运行预检。环境缺失时使用 OneScience CFD 安装入口：

```bash
bash install.sh cfd
```

本示例读取 TFRecord，需要 TensorFlow；训练和推理需要 PyTorch、DGL、Matplotlib、TQDM、ruamel.yaml、PyYAML 等依赖。

## 运行流程

预检：

```bash
python scripts/preflight_vortex_shedding_mgn.py --repo-root . --data-dir data/cylinder_flow
```

快速预检：

```bash
python scripts/preflight_vortex_shedding_mgn.py --repo-root . --data-dir data/cylinder_flow --skip-tfrecord-read
```

单卡训练：

```bash
python train.py
```

多卡训练：

```bash
mpirun -np <num_GPUs> python train.py
```

推理和可视化：

```bash
python inference.py
```

推理前 `checkpoints/` 必须包含训练得到的兼容 checkpoint。推理脚本会在 `animations/` 下生成 `animation_u.gif`、`animation_v.gif` 和 `animation_p.gif`。

## 预检与诊断

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `missing data directory` | 数据集未放到模型包根目录 | 下载 `OneScience/cylinder_flow`，并复制 `data/cylinder_flow` 到模型包 `data/cylinder_flow` |
| `Normalization stats not found` | 缺少 `stats/edge_stats.json` 或 `stats/node_stats.json` | 重新下载数据集，确认 stats 文件存在 |
| `TensorFlow is required for reading .tfrecord files` | 环境缺少 TensorFlow | 安装 OneScience CFD 依赖或安装 TensorFlow |
| `此 DGL 版本的 Datapipe 需要 DGL 库` | 环境缺少 DGL | 安装与 PyTorch/CUDA 匹配的 DGL |
| 推理无法加载 checkpoint | 尚未训练或 checkpoint 路径为空 | 先运行训练命令，或把兼容 checkpoint 放入 `checkpoints/` |

## 输出说明

| 输出 | 路径 | 说明 |
|---|---|---|
| checkpoint | `checkpoints/` | 训练保存的模型、优化器和调度器状态 |
| GIF 动画 | `animations/animation_u.gif` | 水平速度预测与真值对比 |
| GIF 动画 | `animations/animation_v.gif` | 垂直速度预测与真值对比 |
| GIF 动画 | `animations/animation_p.gif` | 压力预测与真值对比 |

## 限制与适用范围

本标准包默认使用 `OneScience/cylinder_flow` 中的 DeepMind CylinderFlow 数据，配置为训练 400 条轨迹、验证 10 条轨迹、测试 10 条轨迹，每条使用 300 个时间步。原始数据 meta 中轨迹长度为 600；当前配置未超过数据可用范围。仓库不包含预训练 checkpoint，推理需要先训练或外部提供 checkpoint。

## 引用与许可证

请引用 MeshGraphNet / Learning Mesh-Based Simulation with Graph Networks 相关工作以及 OneScience 项目。当前整理包的许可证字段暂记为 unknown；使用数据和代码前请遵循原始数据集、论文和 OneScience 仓库的许可要求。
