# XiHe

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 XiHe 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/XiHe" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

XiHe（羲和）是面向高分辨率全球海洋预报的 Transformer 模型运行包，输入为 CMEMS HDF5 网格场，包含海表高度、海表温度、10m 风场以及多层海温、盐度和流速变量，输出为测试时刻的海洋状态预测数组。

本仓库已整理为 OneScience ModelScope 标准运行包，提供配置、训练、推理、评测、可视化、数据准备和预检脚本。当前包与 `OneScience/CMEMS` 数据集配套使用，数据年份为 1993-1999 年；原始配置中的 2010、2013、2014 年已在标准包配置中改为当前数据实际存在的年份。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | earth |
| 领域标签 | earth, ocean, cmems, forecasting |
| 任务 | ocean_forecasting |
| 任务标签 | ocean_forecast, spatiotemporal_forecast, transformer |
| 主平台资源 | https://modelscope.cn/models/OneScience/XiHe |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | 无，当前为标准运行包优先 |
| 必需模型文件 | `conf/config.yaml`, `train.py`, `inference.py`, `result.py`, `scripts/*.py` |
| 必需数据集 | `OneScience/CMEMS` |
| 支持能力 | 预检、训练、推理、评测、可视化 |
| 最小验证 | `python scripts/preflight.py --config conf/config.yaml` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和输出 | 是 | 全部能力 | `XiHe/manifest.yaml` | 默认 Manifest |
| `onescience_run_manifest.yaml` | Manifest 兼容副本 | 供网页端按任务描述读取的兼容入口 | 是 | 全部能力 | `XiHe/onescience_run_manifest.yaml` | 内容与 `manifest.yaml` 一致 |
| `conf/config.yaml` | 配置文件 | 已改写为 1993-1999 年 CMEMS 数据划分 | 是 | 预检、训练、推理、评测 | `XiHe/conf/config.yaml` | 配置改动记录见 Manifest |
| `scripts/prepare_runtime_data.py` | 数据准备脚本 | 将 `OneScience/CMEMS` 下载结果整理到 `./data/data/`，并提取统计量、生成掩码 | 是 | 预检、训练、推理 | `XiHe/scripts/prepare_runtime_data.py` | 运行前先执行 |
| `scripts/preflight.py` | 预检脚本 | 检查配置、年份、HDF5 结构、变量、统计量和掩码 | 是 | 预检 | `XiHe/scripts/preflight.py` | 不依赖 OneScience 模型代码 |
| `train.py` | 训练脚本 | 训练 XiHe 并输出 checkpoint 和 loss | 是 | 训练 | `XiHe/train.py` | 需要 OneScience earth 环境 |
| `inference.py` | 推理脚本 | 使用 `data/checkpoints/model_bak.pth` 对测试集推理 | 是 | 推理 | `XiHe/inference.py` | 需要先训练或放入 checkpoint |
| `result.py` | 评测与可视化脚本 | 计算 RMSE/ACC 并生成图片 | 是 | 评测、可视化 | `XiHe/result.py` | 使用推理输出 |
| `work_slurm.sh` | 集群脚本 | SLURM/DCU 训练入口 | 否 | 训练 | `XiHe/work_slurm.sh` | 需按集群环境调整 |

## Manifest

机器可读 Manifest 位于仓库根目录：`manifest.yaml`。为兼容网页端读取，本仓库也提供内容一致的 `onescience_run_manifest.yaml`。修改任何配置、命令、文件布局或数据关系时，必须同步更新 README、`manifest.yaml` 和 `onescience_run_manifest.yaml`。

## 模型 vs 数据集关系

XiHe 必须配合 `OneScience/CMEMS` 使用。模型 Manifest 的 `relations.required_datasets` 指向 `OneScience/CMEMS`，数据集 Manifest 的 `relations.compatible_models` 反向指向 `OneScience/XiHe`。运行时应先下载两个仓库，再在 XiHe 工作目录中执行数据准备脚本。

## 文件与下载

推荐下载到同一个会话工作目录：

```bash
modelscope download --model OneScience/XiHe --local_dir ./XiHe
modelscope download --dataset OneScience/CMEMS --local_dir ./CMEMS
```

如果使用 `--cache_dir` 下载，ModelScope 可能在缓存目录下创建实际仓库子目录。后续运行的 `cwd` 必须切换到实际下载后的 XiHe 包根目录，也就是包含 `conf/config.yaml`、`train.py` 和 `manifest.yaml` 的目录。

## 环境安装

网站环境已部署 OneScience 时，优先直接运行预检。环境缺失时，按 OneScience earth 领域安装：

```bash
bash install.sh earth
```

本包预检依赖 `python`, `h5py`, `numpy`, `pyyaml`；训练、推理和评测还需要 OneScience earth 环境、`torch` 和可用 GPU/DCU。

## 运行流程

在会话工作目录中下载模型和数据后，执行：

```bash
cd XiHe
python scripts/prepare_runtime_data.py --dataset-root ../CMEMS --runtime-data-dir ./data
python scripts/preflight.py --config conf/config.yaml
```

训练：

```bash
python train.py
```

推理要求存在 `data/checkpoints/model_bak.pth`：

```bash
python inference.py
```

评测和可视化：

```bash
python result.py
```

## 预检与诊断

`scripts/preflight.py` 会检查 `conf/config.yaml` 中的年份是否为 1993-1999，检查 `data/data/{year}.h5` 是否存在，确认 `fields` shape 为 `[3, 96, 2041, 4320]`、dtype 为 `float32`，确认 96 个变量均可在 HDF5 attrs 中找到，并检查 `data/stats/global_means.npy`、`data/stats/global_stds.npy` 和 `data/land_mask.npy`。

常见错误：

| 错误现象 | 原因 | 处理方式 |
|---|---|---|
| `missing yearly HDF5 file` | 数据集未下载或未执行准备脚本 | 先下载 `OneScience/CMEMS`，再执行 `prepare_runtime_data.py` |
| `missing extracted stats file` | 未从 HDF5 提取统计量 | 执行 `python scripts/prepare_runtime_data.py --dataset-root ../CMEMS --runtime-data-dir ./data` |
| `model_bak.pth` 不存在 | 当前仓库不含预训练 checkpoint | 先训练生成 checkpoint，或放入兼容 checkpoint |
| CUDA 相关报错 | 推理脚本默认使用 `cuda:0` | 在 GPU/DCU 环境运行，或按本地设备修改脚本 |

## 输出说明

训练输出位于 `data/checkpoints/`，包括 `model_bak.pth`、`trloss.npy` 和 `valoss.npy`。推理输出位于 `result/output/*.npy`。评测输出包括 `result/rmse.npy`、`result/acc.npy`、`result/loss.png` 和变量对比图。

## 限制与适用范围

当前标准包只适配 `OneScience/CMEMS` 中 1993-1999 年的 7 个 HDF5 文件。仓库不包含预训练权重，推理前需要先训练或提供兼容 checkpoint。`inference.py` 默认使用 GPU `cuda:0`，CPU 推理需要额外修改。

## 引用与许可证

代码许可证沿用原始 XiHe 示例的 Apache 2.0。数据许可证请以 `OneScience/CMEMS` 数据集仓库说明为准。
