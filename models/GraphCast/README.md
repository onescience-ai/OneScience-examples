# GraphCast

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 GraphCast 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/GraphCast/" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

GraphCast 是 Google DeepMind 提出的基于图神经网络的全球中期天气预报模型，通过编码器、消息传递处理器和解码器在规则经纬度网格与多尺度球面网格之间传递信息。它属于 OneScience 的 earth 领域，典型输入是 ERA5 再分析数据中的多变量全球网格场，输出是未来时刻的全球气象变量预测场。

本仓库整理为 OneScience 标准模型运行包，包含 GraphCast 的训练、微调、推理、评测可视化脚本、配置文件、预检脚本和机器可读 Manifest。模型运行依赖数据集 `OneScience/ERA5`；本次按任务要求只整理模型仓库，不复制、不重打包、不生成 ERA5 数据集标准仓库。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| ModelScope 模型 ID | `OneScience/GraphCast/` |
| OneScience 领域 | earth |
| 领域标签 | weather_forecast, era5, graph_neural_network, global_forecasting |
| 任务 | 全球天气预报 |
| 任务标签 | train, finetune, inference, evaluate, visualize |
| 主平台资源 | https://modelscope.cn/models/OneScience/GraphCast/ |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | 无，按标准运行包直接运行 |
| 必需模型文件 | `conf/config.yaml`, `train.py`, `finetune.py`, `inference.py`, `result.py`, `compute_time_diff_std.py`, `get_data_json.py`, `scripts/preflight_model.py` |
| 必需数据集 | `OneScience/ERA5` |
| 支持能力 | 训练、微调、推理、评测、可视化、预检 |
| 最小验证 | `python scripts/preflight_model.py --config conf/config.yaml` |

能力和命令要求

| 能力 | 必须提供 |
|---|---|
| `inference` | `python inference.py`；需要 `data/checkpoints/model_finetune_bak.pth`、`OneScience/ERA5` 测试年份数据和静态文件 |
| `train` | `python train.py`；需要 `OneScience/ERA5` 训练/验证年份数据、静态文件、`data.json` 和 `time_diff_std.npy` |
| `finetune` | `python finetune.py`；需要训练生成的 `data/checkpoints/model_bak.pth` |
| `evaluate` | `python result.py`；需要推理输出、测试年份数据和训练损失文件 |
| `visualize` | `python result.py`；输出 `result/*.png` |
| `deploy` | 当前标准包未声明部署能力 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明模型文件、ERA5 数据集关系、下载方式、命令和输出 | 是 | 全部能力 | `session_workdir/onescience_run_manifest.yaml` | 本次任务指定的主 Manifest |
| `manifest.yaml` | Manifest 文件 | 与 `onescience_run_manifest.yaml` 内容一致的兼容副本，便于按标准默认路径发现 | 是 | 全部能力 | `session_workdir/manifest.yaml` | 修改运行包时必须同步更新两个 Manifest |
| `conf/config.yaml` | 配置文件 | 控制 GraphCast 模型结构、训练轮数、数据路径、年份、变量通道、静态文件、分布式参数和 checkpoint 目录 | 是 | 预检、训练、微调、推理、评测 | `session_workdir/conf/config.yaml` | 已适配为最小验证配置 |
| `scripts/preflight_model.py` | 预检脚本 | 检查配置、数据路径、必需数据文件、变量/schema、静态文件、`data.json` 和 `time_diff_std.npy` | 是 | 预检 | `session_workdir/scripts/preflight_model.py` | 不复制数据；加 `--check-data` 时读取已下载 ERA5 |
| `scripts/validate_standard_package.py` | 校验脚本 | 检查 README 章节、Manifest YAML、ID 一致性、relations 和 command_refs | 是 | 上传前校验 | `session_workdir/scripts/validate_standard_package.py` | 本地标准包校验辅助脚本 |
| `train.py` | 运行脚本 | 训练 GraphCast，并保存 `model_bak.pth` 与训练损失 | 是 | 训练 | `session_workdir/train.py` | 单卡命令为 `python train.py` |
| `finetune.py` | 运行脚本 | 基于 `model_bak.pth` 执行多步 rollout 微调 | 是 | 微调 | `session_workdir/finetune.py` | 输出 `model_finetune_bak.pth` |
| `inference.py` | 运行脚本 | 加载 `model_finetune_bak.pth`，对测试年份 ERA5 数据执行推理 | 是 | 推理 | `session_workdir/inference.py` | 输出到 `result/output/` |
| `result.py` | 运行脚本 | 计算 RMSE/ACC，并绘制预测、真值和误差图 | 是 | 评测、可视化 | `session_workdir/result.py` | 默认绘制测试年份第一个样例 |
| `compute_time_diff_std.py` | 准备脚本 | 从训练数据计算 GraphCast 损失函数需要的 `time_diff_std.npy` | 是 | 数据准备、训练 | `session_workdir/compute_time_diff_std.py` | 需要已下载或挂载 ERA5 |
| `get_data_json.py` | 准备脚本 | 根据配置通道生成 `data.json` 变量元数据 | 是 | 数据准备、训练 | `session_workdir/get_data_json.py` | 运行成本低，不复制数据 |
| `fake_data.py` | 辅助脚本 | 生成本地虚拟 HDF5 与静态文件，用于快速流程 smoke test | 否 | 最小流程验证 | `session_workdir/fake_data.py` | 虚拟数据不属于本次整理的数据集上传内容 |
| `work_slurm.sh` | 集群脚本 | 在 Slurm/DCU 环境提交分布式训练任务 | 否 | 训练 | `session_workdir/work_slurm.sh` | 使用前需按集群分区和环境修改 |
| `data/` | 外部数据目录 | 放置从 `OneScience/ERA5` 下载或挂载的 ERA5 HDF5、静态文件、派生统计文件和 checkpoint 输出 | 是 | 训练、微调、推理、评测 | `session_workdir/data/` | 本仓库不上传数据文件 |

## Manifest

主 Manifest 文件为仓库根目录下的 `onescience_run_manifest.yaml`。为兼容标准默认发现路径，仓库同时提供内容一致的 `manifest.yaml`。修改文件路径、数据集 ID、下载命令、配置适配、运行命令或输出路径后，必须同步更新两个 Manifest，并重新执行 YAML 解析、command_refs 校验和模型预检。

## 模型 vs 数据集关系

GraphCast 模型依赖 ERA5 HDF5 数据集，数据集 ID 固定为 `OneScience/ERA5`。模型侧 Manifest 在 `relations.required_datasets` 中声明该数据集，并在 `run_matrix` 中说明配置预检、数据预检、数据派生文件准备、训练、微调、推理、评测和可视化分别需要哪些文件与命令。

本次只处理模型标准仓库，不整理数据集文件，也不生成数据集标准仓库。ERA5 数据应从 `OneScience/ERA5` 下载或由 OneScience 环境挂载到 `session_workdir/data/`，目录下至少应包含 `data/<year>.h5` 和 `static/`；HDF5 文件需包含 `fields`、`global_means`、`global_stds`，且 `fields.attrs` 包含 `variables` 和 `time_step`。GraphCast 还需要在模型包根目录生成 `data.json` 和 `time_diff_std.npy`。

## 文件与下载

下载模型标准包：

```bash
modelscope download --model OneScience/GraphCast/ --local_dir ./earth_graphcast
cd ./earth_graphcast
```

下载 ERA5 数据集到模型默认数据目录：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

如果网页端或运行环境使用 `--cache_dir` 下载模型，请在下载完成后把当前工作目录切换到实际模型包根目录，也就是包含 `onescience_run_manifest.yaml`、`conf/config.yaml` 和 `train.py` 的目录，再执行预检和运行命令。

## 环境安装

OneScience 网站环境通常已部署运行依赖。若本地环境缺少依赖，可在 OneScience 主仓库中安装 earth 领域：

```bash
bash install.sh earth
```

模型脚本需要 Python 3.11、PyTorch、OneScience、h5py、numpy、xarray、matplotlib、tqdm、PyYAML、ruamel.yaml；训练脚本还使用 Apex `FusedAdam`。如果使用本机 conda，可先激活 `py311` 环境：

```bash
source /Users/zhao/Desktop/MiniConda/miniconda3/bin/activate py311
```

## 运行流程

### 1. 环境预检

```bash
python -c "import torch, h5py, yaml; import onescience; print('environment ok')"
```

### 2. 下载

```bash
modelscope download --model OneScience/GraphCast/ --local_dir ./earth_graphcast
cd ./earth_graphcast
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

### 3. 应用运行包和准备文件

模型下载后已经是标准运行包。真实运行时请确认 `conf/config.yaml` 中的 `datapipe.dataset.data_dir` 指向 `./data/` 或实际挂载的 ERA5 数据目录，`datapipe.dataset.static_dir` 指向静态文件目录。

准备 GraphCast 专用派生文件：

```bash
python get_data_json.py
python compute_time_diff_std.py
```

如只做流程 smoke test，可在模型目录生成虚拟数据：

```bash
python fake_data.py
python get_data_json.py
python compute_time_diff_std.py
```

### 4. 运行前预检

只检查模型配置：

```bash
python scripts/preflight_model.py --config conf/config.yaml
```

同时检查已下载 ERA5 数据结构、静态文件和派生统计文件：

```bash
python scripts/preflight_model.py --config conf/config.yaml --check-data
```

### 5. 运行

训练：

```bash
python train.py
```

微调：

```bash
python finetune.py
```

推理：

```bash
python inference.py
```

评测与可视化：

```bash
python result.py
```

### 6. 验证输出

训练成功后应生成 `data/checkpoints/model_bak.pth` 和 `data/checkpoints/trloss.npy`。微调成功后应生成 `data/checkpoints/model_finetune_bak.pth` 和 `data/checkpoints/ft_trloss.npy`。推理成功后应生成 `result/output/*.npy`。评测和可视化成功后应生成 `result/rmse.npy`、`result/acc.npy`、`result/loss.png` 和若干 `result/*.png`。

## 预检与诊断

| 错误现象 | 可能原因 | 处理方式 |
|---|---|---|
| `ModuleNotFoundError: No module named 'onescience'` | OneScience earth 环境未安装或未激活 | 在 OneScience 主仓库执行 `bash install.sh earth`，或切换到网站提供的 OneScience 环境 |
| `ModuleNotFoundError: No module named 'apex'` | 训练依赖 Apex 不存在 | 安装与当前 PyTorch/硬件匹配的 Apex，或按 OneScience earth 环境重新部署 |
| `未找到数据文件: ./data/data/*.h5` | ERA5 数据未下载到配置指定目录 | 执行 `modelscope download --dataset OneScience/ERA5 --local_dir ./data`，或修改 `conf/config.yaml` 的 `datapipe.dataset.data_dir` |
| `GraphCast 静态文件目录不存在` | 缺少 `data/static/` | 从 `OneScience/ERA5` 下载静态文件，或运行 `python fake_data.py` 做本地 smoke test |
| `data.json` 不存在 | 尚未生成变量元数据 | 在模型包根目录执行 `python get_data_json.py` |
| `time_diff_std.npy` 不存在 | 尚未计算 GraphCast 归一化时间差统计量 | 准备训练年份数据后执行 `python compute_time_diff_std.py` |
| `Variables not found` 或预检提示缺少配置通道 | HDF5 `fields.attrs.variables` 与配置通道不匹配 | 使用与 GraphCast 配置匹配的 ERA5 数据，或同步修改 `conf/config.yaml` 和 Manifest 的配置适配说明 |
| `model_bak.pth` 不存在 | 尚未训练基础模型 | 先执行 `python train.py`，或放置兼容 checkpoint 到 `data/checkpoints/model_bak.pth` |
| `model_finetune_bak.pth` 不存在 | 尚未完成微调，推理脚本默认加载微调 checkpoint | 先执行 `python finetune.py`，或将兼容 checkpoint 放到 `data/checkpoints/model_finetune_bak.pth` |
| `CUDA out of memory` | 显存不足或 batch size 过大 | 降低 `conf/config.yaml` 中 `datapipe.dataloader.batch_size`，或切换更大显存设备 |
| `modelscope: command not found` | 未安装 ModelScope CLI | 安装并登录 ModelScope CLI，或通过网页下载后保持相同目录结构 |

## 输出说明

| 输出路径 | 类型 | 产生阶段 | 说明 |
|---|---|---|---|
| `data/checkpoints/model_bak.pth` | checkpoint | 训练 | GraphCast 基础模型权重和优化器状态 |
| `data/checkpoints/model_finetune_bak.pth` | checkpoint | 微调 | GraphCast 多步 rollout 微调权重和优化器状态 |
| `data/checkpoints/trloss.npy` | numpy 数组 | 训练 | 训练损失序列 |
| `data/checkpoints/ft_trloss.npy` | numpy 数组 | 微调 | 微调损失序列 |
| `data.json` | JSON | 数据准备 | GraphCast 损失函数需要的通道元数据 |
| `time_diff_std.npy` | numpy 数组 | 数据准备 | GraphCast 损失函数需要的时间差标准差 |
| `result/output/*.npy` | numpy 数组 | 推理 | 每个测试时刻的预测气象场 |
| `result/rmse.npy` | numpy 数组 | 评测 | 各变量 RMSE |
| `result/acc.npy` | numpy 数组 | 评测 | 各变量 ACC |
| `result/*.png` | 图片 | 可视化 | 损失曲线、真值、预测和误差图 |

## 限制与适用范围

本标准包默认配置面向最小验证：`max_epoch`、`num_iters_step1`、`num_iters_step2`、`num_iters_step3` 均调整为 1，训练、验证、测试年份分别为 1951、1952、1953，分布式开关关闭。正式训练或 benchmark 前，应根据 `OneScience/ERA5` 实际可用年份、硬件规模和实验目标调整配置，并同步更新 Manifest 中的 `configuration_adaptation`。

本仓库不包含预训练 checkpoint。直接推理需要先训练并微调生成 `data/checkpoints/model_finetune_bak.pth`，或由用户放置兼容 checkpoint。虚拟数据只用于流程验证，不能代表真实 ERA5 评测效果。

## 引用与许可证

论文：GraphCast: Learning skillful medium-range global weather forecasting, https://arxiv.org/abs/2212.12794

许可证：Apache 2.0。请同时遵守 ERA5 数据集和 ModelScope 平台的使用条款。
