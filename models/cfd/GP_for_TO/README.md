# GP_for_TO

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 GP_for_TO 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/GP_for_TO" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

GP_for_TO 是基于物理信息高斯过程的 CFD 拓扑优化标准运行包，整理自 `onescience/examples/cfd/GP_for_TO/`。模型使用共享神经网络均值函数和多个独立高斯过程输出，联合表示速度 `u`、速度 `v`、压力 `p` 和材料密度 `ro`，通过 PDE 残差、耗散功率和体积约束同时优化二维 Stokes/Brinkman 流设计。

本模型支持 `doublepipe`、`diffuser`、`rugby` 和 `pipebend` 四类设计问题。输入不是外部数据文件，而是命令行选择的问题类型；运行时由 OneScience 的 `onescience.utils.GP_TO.get_data_fluid` 生成配点、边界条件样本和状态变量训练目标。输出包括训练日志、损失曲线、密度场、速度场和残差可视化图。

本仓库已按 OneScience ModelScope 运行标准整理为模型运行包，包含训练入口、优化循环、Slurm 示例、机器可读 Manifest 和预检脚本。由于配套数据集未开源且该示例可在运行时生成训练张量，本仓库不上传数据集目录，也不要求下载外部数据集。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, topology_optimization, gaussian_process, physics_informed_learning, stokes_flow |
| 任务 | physics_informed_topology_optimization |
| 任务标签 | train, optimization, visualization, meshfree_method |
| 主平台资源 | https://modelscope.cn/models/OneScience/GP_for_TO |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/GP_for_TO` |
| 必需模型文件 | `main_TO.py`, `train.py`, `scripts/preflight_gp_for_to.py` |
| 必需数据集 | 无；训练样本由 `onescience.utils.GP_TO.get_data_fluid` 运行时生成 |
| 支持能力 | 预检、训练、验证、推理、预测可视化 |
| 最小验证 | `python scripts/preflight_gp_for_to.py --repo-root .` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、命令、输出和诊断规则 | 是 | 全部能力 | 模型包根目录 | 修改运行包时必须同步更新 |
| `onescience_relations.yaml` | 关系索引 | 声明本模型不依赖外部数据集 | 是 | 自动索引 | 模型包根目录 | 与 Manifest relations 保持一致 |
| `main_TO.py` | 运行脚本 | 生成训练张量、构建 GPPLUS 模型并启动拓扑优化 | 是 | 训练、可视化 | 模型包根目录 | 支持 `--problem` 和 `--gpu` |
| `train.py` | 训练辅助脚本 | 定义 PDE 残差、动态权重、损失函数、优化循环和绘图 checkpoint | 是 | 训练、可视化 | 模型包根目录 | 被 `main_TO.py` 导入 |
| `scripts/preflight_gp_for_to.py` | 预检脚本 | 检查文件、入口参数、训练设置和无外部数据集约束；可选检查运行依赖 | 是 | 预检 | `scripts/preflight_gp_for_to.py` | `--check-imports` 会导入 torch/gpytorch/OneScience |
| `slurm.sh` | 示例脚本 | Slurm/DCU 集群运行示例 | 否 | 训练 | 模型包根目录 | 需按集群环境修改模块和 conda 路径 |

## Manifest

机器可读 Manifest 位于仓库根目录：

```text
onescience_run_manifest.yaml
```

大模型应先读取本 README 的“文件说明”和本节，然后解析 `onescience_run_manifest.yaml`。修改文件、命令、输出、诊断或数据策略时，必须同步更新 Manifest，并重新运行 YAML 解析和 command_refs 校验。

## 模型 vs 数据集关系

本模型没有外部必需数据集。`main_TO.py` 会调用 `onescience.utils.GP_TO.get_data_fluid(problem=..., N_col_domain=10000, N_train=25)`，在运行时生成：

| 内容 | 说明 |
|---|---|
| 配点 | 单位方形设计域上的 `10000 x 2` collocation 坐标 |
| 边界训练坐标 | 每个问题的入口、出口、壁面和角点坐标 |
| 状态变量训练目标 | `u`, `v`, `p`, `ro` 四个输出对应的边界或约束目标 |
| 支持问题 | `doublepipe`, `diffuser`, `rugby`, `pipebend` |

因此 Manifest 中 `relations.required_datasets` 为空，`relations.data_policy.external_dataset_required` 为 `false`。如果后续公开了复现实验数据或预训练权重，应新增独立数据集仓库，并在模型 Manifest 和数据集 Manifest 中补充双向关系。

## 文件与下载

下载模型包：

```bash
modelscope download --model OneScience/GP_for_TO
```

如果使用 `modelscope download --cache_dir`，命令完成后请切换到实际下载后的模型包根目录再执行预检或训练命令。由于本模型不需要外部数据集，本仓库没有 `modelscope download --dataset ...` 的必需步骤。

## 环境安装

OneScience 网站环境如果已经部署 CFD 依赖，可直接运行结构预检。完整训练需要 OneScience CFD 环境、PyTorch、GPyTorch、NumPy、Matplotlib 和 TQDM。环境缺失时使用 OneScience CFD 安装入口：

```bash
bash install.sh cfd
```

如果 OneScience 没有安装为 Python 包，但本地有源码，可以在完整依赖预检或训练前设置：

```bash
export ONESCIENCE_SRC=/path/to/onescience/src
export PYTHONPATH="$ONESCIENCE_SRC:$PYTHONPATH"
```

在 HOME 不可写的集群环境中，建议设置 Matplotlib 缓存目录：

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache-$USER
```

## 运行流程

结构预检，不导入重型训练依赖：

```bash
python scripts/preflight_gp_for_to.py --repo-root .
```

完整依赖和生成数据 schema 预检：

```bash
python scripts/preflight_gp_for_to.py --repo-root . --problem doublepipe --check-imports
```

默认问题训练：

```bash
python main_TO.py --problem doublepipe --gpu 0
```

选择其他拓扑优化问题：

```bash
python main_TO.py --problem diffuser --gpu 0
python main_TO.py --problem rugby --gpu 0
python main_TO.py --problem pipebend --gpu 0
```

`--problem` 只能取 `doublepipe`、`diffuser`、`rugby`、`pipebend`。`--gpu` 是 CUDA 设备编号；没有可用 GPU 时脚本会回退到 CPU，但完整训练耗时会显著增加。

## 预检与诊断

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `runtime dependency import failed for torch` | 未安装 PyTorch 或未激活 OneScience CFD 环境 | 执行 `bash install.sh cfd` 或激活包含 PyTorch 的环境 |
| `runtime dependency import failed for gpytorch` | 缺少 GPyTorch | 安装与当前 PyTorch 兼容的 `gpytorch` |
| `OneScience GP_for_TO imports failed` | OneScience 未安装，或 `PYTHONPATH` 未包含 `onescience/src` | 设置 `ONESCIENCE_SRC` 和 `PYTHONPATH`，或安装 OneScience |
| `unexpected --problem choices` | `main_TO.py` 被修改但 Manifest 未同步 | 恢复四个问题选项，或同步更新 README 和 Manifest |
| `CUDA out of memory` | 配点数、网络宽度或迭代过程显存需求较高 | 使用更大显存设备；如需改小参数，必须同步记录配置变更 |
| Matplotlib 缓存目录警告 | HOME 下配置目录不可写 | 设置 `MPLCONFIGDIR=/tmp/matplotlib-cache-$USER` |

## 输出说明

| 输出 | 路径 | 说明 |
|---|---|---|
| 预检结果 | `stdout` | 成功时输出 `[OK] model preflight completed` |
| 训练日志 | `stdout` | TQDM 进度条输出 loss，训练结束输出总耗时 |
| 可视化图像 | `./*.png` 或可视化函数生成的相对路径 | 在 1000、10000、20000、30000、40000、50000 迭代处生成损失、密度、速度和残差图 |

## 限制与适用范围

本仓库不包含预训练 checkpoint，也不包含外部数据集。完整训练默认 `N_col_domain=10000`、每条边界 `N_train_per_BC=25`、`num_iter=50000`，适合 GPU 或集群环境运行；结构预检可在无 PyTorch 的轻量环境中执行。

当前标准包没有修改原始模型目录，也没有改写训练参数。若后续为了快速 demo 调低迭代数、配点数或网络宽度，需要在 `onescience_run_manifest.yaml` 的 `configuration_adaptation` 中记录原值、现值和修改原因。

## 引用与许可证

请引用 GP_for_TO 原始论文和 OneScience 项目：

- Simultaneous and Meshfree Topology Optimization with Physics-informed Gaussian Processes
- OneScience: https://gitee.com/onescience-ai/onescience

当前整理包许可证字段暂记为 `unknown`；使用代码和论文实现前请遵循原始项目、论文和 OneScience 仓库的许可要求。
