

<p align="center">
  <strong><span style="font-size: 30px;">Aardvark Weather</span></strong>
</p>

# 模型介绍

Aardvark Weather 是端到端多模态天气预报模型，通过观测编码器、全球预报处理器和站点解码器生成全球格点及站点天气预报。

论文：End-to-end data-driven weather prediction

https://www.nature.com/articles/s41586-025-08897-0

# 模型描述

当前模型包复用官方代码和权重，提供以下运行链路：

```text
官方多模态样例
-> Encoder
-> Day-1 Processor
-> TAS Decoder
-> 1 天全球预报和站点 2 米温度
```

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 官方模型验证 | 检查官方样例、配置和 checkpoint。 |
| 全球天气预报 | 输出 24 个变量的全球 1.5° 格点状态。 |
| 站点温度预报 | 输出 8,719 个站点的 2 米温度。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `download.sh` | ModelScope 资源下载脚本 | 恢复官方样例、归一化资源和已发布 checkpoint |
| `conf/config.yaml` | 训练配置 | 定义数据路径、训练轮数、优化器参数和可训练模块 |
| `scripts/train.py` | 训练入口 | 支持 Decoder 微调、联合微调和断点续训 |
| `scripts/inference.py` | 一天预报推理入口 | 加载官方资源或本地微调 checkpoint |
| `scripts/result.py` | 结果评估与可视化入口 | 输出归一化 RMSE、MAE 和站点温度对比图 |
| `model/aardvark_adapter.py` | 官方模型适配层 | 校验样例与 checkpoint，并组装一天预报链路 |
| `model/sample_dataset.py` | 样例数据集适配 | 负责样例发现、训练/验证划分和 batch 拼接 |
| `official-src/` | Aardvark 官方代码快照 | 包含模型实现、notebook 和官方训练脚本 |
| `weights/` | 官方样例与预训练资源目录 | 资源由 `download.sh` 下载到既定相对路径 |
| `weight/` | 本地训练权重目录 | 保存训练产生的 checkpoint |

# 使用说明

## 1. OneCode 使用

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 当前官方权重推理需要 NVIDIA GPU。
- CPU 可用于资源和 checkpoint 检查，不建议运行完整推理。

## 3. 快速开始
### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

如环境缺少 Aardvark 依赖，请参考 `official-src/environment.yml` 补充安装。当前适配已兼容新版 `timm` 的 `Block` 参数。

### 数据与权重

模型包内已经包含 1 天温度推理所需资源：

```text
weights/sample_data/sample_data_final.pkl
weights/trained_model/encoder/epoch_96
weights/trained_model/processor/forecast_1/epoch_0
weights/trained_model/decoder/tas/lt_1/epoch_18
official-src/data/grid_lon_lat/
official-src/data/norm_factors/
```

如需重新下载，可使用：

```text
官方代码：https://github.com/anna-allen/aardvark-weather-public
官方权重：https://huggingface.co/datasets/av555/aardvark-weather
```

### 训练

训练入口提供完整的 epoch、验证、early stopping、学习率调度、最佳/最新 checkpoint 和断点续训流程。默认冻结 Encoder 和 Day-1 Processor，只训练 TAS Decoder：

```bash
python scripts/train.py
```

使用论文式端到端联合微调：

```bash
python scripts/train.py --train-modules all --epochs 10 --train-steps 100
```

恢复训练：

```bash
python scripts/train.py --resume weight/training/last.pth
```

默认配置位于 `conf/config.yaml`。`--data` 可以指向一个官方 schema pickle，也可以指向包含多个 `.pkl` 的目录；多文件会稳定划分训练集和验证集。`--batch-size` 会沿官方 task 的现有 batch 维拼接多个任务。随包只有一个官方样例时，训练和验证会重复使用同一任务，因此能够完整验证训练软件链路，但不构成独立验证集，也不能提供论文数据多样性或复现论文精度。

训练产物：

```text
weight/training/best.pth
weight/training/last.pth
weight/training/history.json
weight/training/train.json
```

### 推理

```bash
python scripts/inference.py
```

使用训练得到的权重：

```bash
python scripts/inference.py --checkpoint weight/training/best.pth
```

推理默认加载官方 sample 和 1 天 `tas` 权重。结果报告保存到：

```text
result/inference_one_day.json
result/prediction.pt
result/target.pt
```

### 结果检验

```bash
python scripts/result.py
```

已验证的输出为：

```text
initial_state:   [1, 121, 240, 24]
global_forecast: [1, 121, 240, 24]
station_tas:     [1, 8719]
```

当前结果属于运行连通性验证，不包含论文 RMSE/MAE 复现。

结果脚本另外生成 `result/metrics.json` 和 `result/comparison.png`。其中 `normalized_mae` 和 `normalized_rmse` 在官方样例的归一化空间计算，不可与论文物理单位指标直接比较。

### 论文与当前实现的 I/O

| 项目 | 论文 | 当前模型包 |
| --- | --- | --- |
| 输入 | 多模态卫星、站点、船舶和探空观测 | 随包官方样例 pickle，字段结构与官方 Encoder 一致 |
| 全球状态 | `24 x 121 x 240`，1.5 度 | 支持 Day-1，输出 `[1,121,240,24]` |
| 站点输出 | 2 米温度和 10 米风，最长 Day-10 | 仅 Day-1 TAS，`[1,8719]` |
| 训练 | 分阶段预训练及约 25,000 步端到端微调 | 可配置完整训练循环；支持 Decoder 或全模型联合微调 |
| 评估 | 物理单位的加权格点 RMSE 和站点 MAE | 归一化样例空间 MAE/RMSE |

所有命令应从项目根目录运行；`scripts/inference.py --root` 会转换为绝对路径。官方模型内部依赖 CUDA，因此 CPU 当前不可用。训练数据必须保持官方多模态任务字典 schema；当前目录不伪造卫星和站点观测，随包官方样例承担默认训练链路验证。模型包只保留 `weights/` 下的官方资源，不携带 `weight/` 本地训练产物或 `result/` 生成物。论文级训练仍需准备完整日期范围的观测数据并转换为相同 `.pkl` 任务契约。

### 真实数据

接入真实日期需要准备 ASCAT、AMSU-A/B、HIRS、IASI、GridSat、HadISD、ICOADS、IGRA、ERA5、地形、气候态和对应归一化统计量。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 官方代码：https://github.com/anna-allen/aardvark-weather-public
- 本目录为官方 Aardvark Weather 模型的独立适配。
- 代码、权重和数据分别遵循其官方许可证与数据条款。
