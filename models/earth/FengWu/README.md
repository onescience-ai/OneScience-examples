# FengWu

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 FengWu 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/FengWu/" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

FengWu（风乌）是面向全球中期天气预报的深度学习模型运行包，属于 OneScience 的 earth 领域。该标准化仓库提供训练、推理、评测和可视化脚本，输入为 ERA5 年度 HDF5 格点场，输出为预测的多变量大气场、训练 checkpoint、RMSE/ACC 指标和可视化图片。

本仓库按 OneScience ModelScope 大模型运行标准整理，模型上传目标 ID 固定为 `OneScience/FengWu/`。当前包不内置预训练 checkpoint；推理、评测和可视化需要先通过训练生成 `data/checkpoints/model_bak.pth`，或由用户放入兼容 checkpoint。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | earth |
| 领域标签 | earth, atmosphere, weather |
| 任务 | weather_forecast |
| 任务标签 | weather_forecast, spatiotemporal_forecast |
| 主平台资源 | https://modelscope.cn/models/OneScience/FengWu/ |
| ModelScope 模型 ID | `OneScience/FengWu/` |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | 无，当前为标准运行包 |
| 必需模型文件 | `conf/config.yaml`, `train.py`, `inference.py`, `result.py` |
| 必需数据集 | `OneScience/ERA5` |
| 支持能力 | 预检、训练、推理、评测、可视化 |
| 最小验证 | `python scripts/preflight_fengwu.py --workdir .` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文件 | 人类和大模型的入口说明 | 是 | 全部能力 | 仓库根目录 | 正文中文 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、关系、命令和诊断 | 是 | 全部能力 | 仓库根目录 | 主要 Manifest |
| `manifest.yaml` | Manifest 兼容入口 | 兼容标准默认 Manifest 文件名 | 是 | 全部能力 | 仓库根目录 | 内容与主 Manifest 一致 |
| `conf/config.yaml` | 配置文件 | FengWu 模型、ERA5 数据路径、年份、通道和 dataloader 配置 | 是 | 预检、训练、推理、评测 | `conf/config.yaml` | 已适配当前 ERA5 数据年份 |
| `scripts/preflight_fengwu.py` | 预检脚本 | 检查配置、数据路径、年度文件、变量、shape、dtype 和统计量 | 是 | 预检 | `scripts/preflight_fengwu.py` | 不启动训练 |
| `train.py` | 运行脚本 | 训练 FengWu 并写出 checkpoint 和 loss | 是 | 训练 | `train.py` | checkpoint 输出到 `data/checkpoints/` |
| `inference.py` | 运行脚本 | 读取 checkpoint 对测试年份执行推理 | 条件必需 | 推理 | `inference.py` | 需要 `data/checkpoints/model_bak.pth` |
| `result.py` | 运行脚本 | 计算 RMSE/ACC 并绘制结果 | 条件必需 | 评测、可视化 | `result.py` | 需要推理输出 |
| `fake_data.py` | 辅助脚本 | 生成极小 HDF5 测试数据 | 否 | 调试 | `fake_data.py` | 不替代真实 ERA5 |
| `work_slurm.sh` | 集群脚本 | Slurm/DCU 训练提交示例 | 否 | 训练 | `work_slurm.sh` | 提交前检查分区和环境 |

## Manifest

主 Manifest 文件是 `onescience_run_manifest.yaml`，兼容入口是 `manifest.yaml`。大模型应先读取本 README，再解析 Manifest；修改配置、命令或文件结构时必须同步更新这两个 Manifest 文件。

## 模型 vs 数据集关系

本模型必须配套数据集 `OneScience/ERA5`。模型 Manifest 的 `relations.required_datasets` 指向该数据集，数据集 Manifest 的 `relations.compatible_models` 反向指向 `OneScience/FengWu/`。默认配置要求数据集放在模型工作目录的 `data/ERA5/`，其内部应包含 `data/1979.h5` 到 `data/2025.h5`。

## 文件与下载

下载模型运行包：

```bash
modelscope download --model OneScience/FengWu/ --local_dir .
```

下载数据集到模型包期望位置：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir data/ERA5
```

如果网页端或自动工具使用 `--cache_dir` 下载模型，运行前需要把 `cwd` 切换到实际下载后的模型包根目录，也就是包含 `conf/config.yaml` 和 `onescience_run_manifest.yaml` 的目录。

## 环境安装

优先使用网站已有 OneScience 环境。环境缺失时，从 OneScience 主仓库安装 earth 领域依赖：

```bash
bash install.sh earth
```

运行训练和推理需要 PyTorch、h5py、numpy、tqdm、matplotlib，以及 OneScience 的 `onescience.models.fengwu` 和 `onescience.datapipes.climate.ERA5Datapipe`。

## 运行流程

1. 下载模型包并进入模型包根目录。
2. 下载 `OneScience/ERA5` 到 `data/ERA5`。
3. 执行预检：

```bash
python scripts/preflight_fengwu.py --workdir .
```

4. 训练：

```bash
python train.py
```

5. 训练生成 `data/checkpoints/model_bak.pth` 后推理：

```bash
python scripts/preflight_fengwu.py --workdir . --require-checkpoint
python inference.py
```

6. 评测和可视化：

```bash
python result.py
```

## 预检与诊断

预检脚本会检查 `conf/config.yaml` 是否可解析，`datapipe.dataset.data_dir` 是否存在，训练/验证/测试年份文件是否齐全，HDF5 中是否存在 `fields`、`global_means`、`global_stds`，shape、dtype、变量列表和统计量是否匹配。

常见错误：

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `missing ERA5 HDF5 directory` | 数据集未下载到 `data/ERA5` | 运行 `modelscope download --dataset OneScience/ERA5 --local_dir data/ERA5` |
| `missing configured year files` | 年度 HDF5 文件不完整 | 对照数据集 checksum 重新整理或下载 |
| `missing configured channels` | 数据 schema 与配置不匹配 | 使用 `OneScience/ERA5` 标准数据或同步修改配置和 Manifest |
| `missing checkpoint for inference/evaluate` | 尚未训练或 checkpoint 未放置 | 先训练，或将兼容 checkpoint 放入 `data/checkpoints/model_bak.pth` |

## 输出说明

训练输出位于 `data/checkpoints/`，包括 `model_bak.pth`、`trloss.npy` 和 `valoss.npy`。推理输出位于 `result/output/`，评测指标为 `result/rmse.npy` 和 `result/acc.npy`，可视化图片输出到 `result/`。

## 限制与适用范围

本整理包当前不包含预训练权重，默认训练配置面向 1979-2025 年 ERA5 HDF5 数据。全量训练计算量较大，自动化流程默认先运行预检；训练、推理和评测需要具备 OneScience earth 运行环境和可用加速硬件。

## 引用与许可证

FengWu 原始论文：`FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead`。原始模型代码 README 标注许可证为 Apache 2.0；数据许可证以 ERA5 数据来源和 `OneScience/ERA5` 数据集仓库说明为准。
