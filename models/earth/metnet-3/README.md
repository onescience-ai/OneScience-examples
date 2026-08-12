<p align="center">
  <strong><span style="font-size: 30px;">MetNet-3 Compact</span></strong>
</p>

# 模型介绍

MetNet-3 是面向稀疏观测的区域高分辨率天气预报模型，可预测降水、温度、露点和风等变量。

论文：Deep Learning for Day Forecasts from Sparse Observations

https://arxiv.org/abs/2306.06079

# 模型描述

当前目录根据论文实现独立 compact smoke 版本，保留多源输入接口、lead-time conditioning、稀疏 OMO mask、概率输出和 HRRR 辅助回归。输入规模、时间融合、MaxViT 和输出分辨率均经过简化。

当前模型用于功能验证，不是 Google 官方实现，也不提供官方预训练权重。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多源天气模型研究 | 验证 MRMS、OMO、HRRR、GOES 等输入接口。 |
| 本地快速验证 | 使用 fake 数据完成训练、checkpoint 和推理。 |
| 真实区域预报 | 后续可接入真实 MRMS、OMO、HRRR 和 GOES。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `download.sh` | ModelScope 资源下载脚本 | 当前无额外模型或数据文件需要下载 |
| `model/metnet3.py` | Compact MetNet-3 主模型 | 组织多源编码、lead-time conditioning 和多任务输出 |
| `model/metnet3_blocks.py` | 主干网络模块 | 实现条件卷积块和 compact long-range MaxViT |
| `model/metnet3_heads.py` | 多任务输出头 | 提供降水、地面变量分类和 HRRR 回归头 |
| `model/metnet3_losses.py` | 训练损失 | 汇总概率分类与辅助回归目标 |
| `model/metnet3_schema.py` | 输入数据契约 | 定义通道、时间帧、输出 bins 并校验 batch |
| `model/fake_data.py` | 合成多源数据生成模块 | 构造 MRMS、OMO、HRRR、GOES 和静态场输入 |
| `scripts/train.py` | 训练入口 | 执行训练、验证、early stopping 和断点续训 |
| `scripts/inference.py` | 推理入口 | 加载 checkpoint 并生成多任务预测 |
| `scripts/result.py` | 评估与可视化入口 | 输出概率、回归指标和结果图 |
| `weight/` | 本地训练权重目录 | 保存训练产生的 checkpoint |

# 使用说明

## 1. OneCode 使用

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- CPU 可运行当前 compact 配置。
- 真实数据和大配置推荐使用 GPU。

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

### 数据

当前 `model/fake_data.py` 提供可索引 fake Dataset，生成 MRMS、OMO、HRRR proxy、GOES proxy、地形、坐标、时间和 lead-time 数据，不需要额外下载。每个样本去除 batch 维，DataLoader 再按 batch 拼接。

### 训练

```bash
python scripts/train.py
```

脚本执行多 epoch 多任务训练、独立验证、学习率调度和 early stopping：

1. 生成 train/validation fake Dataset 和多任务 targets；
2. 每个 batch 校验 input schema；
3. 计算降水 CE、地面变量 CE 和 HRRR MSE 并更新参数；
4. 在独立验证 Dataset 上计算 validation loss；
5. 保存 latest/best checkpoint、history，并支持 `--resume`；
6. 推理阶段读取 `weight/model.pth`。

```bash
python scripts/train.py --epochs 10
python scripts/train.py --resume weight/training/latest.pth --epochs 20
```

checkpoint 输出：

```text
weight/model.pth
weight/training/latest.pth
weight/training/best.pth
weight/training/history.json
```

### 推理

```bash
python scripts/inference.py
```

推理结果：

```text
result/prediction.pt
result/target.pt
result/inference.json
```

### 结果检验

```bash
python scripts/result.py
```

当前输出来自 fake 数据，只用于检查模型能否运行，不代表论文预报指标。

结果脚本生成 `result/metrics.json` 和 `result/comparison.png`，报告归一化 bin 空间的降水/地面 MAE、HRRR proxy RMSE 和概率归一化误差，不是论文 CRPS、CSI 或物理单位 MAE。

### 论文与当前实现的 I/O

| 输入/输出 | 论文 | 当前 compact 配置 |
| --- | --- | --- |
| MRMS high | 2 通道 x 11 帧 | 4 通道 x 3 帧，仅使用最后一帧 |
| MRMS low | 1 通道 x 1 帧 | 3 通道 x 2 帧，仅使用最后一帧 |
| OMO | 14 通道 x 9 帧 | 2 通道 x 3 帧，仅使用最后一帧 |
| HRRR/GOES | 618 / 16 通道 | 8 / 4 proxy 通道 |
| 降水输出 | 两类目标，各 512 bins | 单目标，16 bins |
| 地面输出 | 6 变量，256/180 bins | 6 变量，统一 8 bins |
| 主干 | 改造版 12-block MaxViT | 单层 Transformer proxy |
| 训练 | 完整数据与多任务训练 | 多 epoch fake Dataset 多任务训练 |

完整执行顺序为 `train.py -> inference.py -> result.py`。fake Dataset 单样本输入为 `[T,C,H,W]`，targets 为对应的多任务字典，默认空间大小 `8x8`；`current_time` 和 `lead_time` 均为 `[1]`，DataLoader 后为 `[B,1]`。checkpoint 固定兼容写入 `weight/model.pth`，训练历史和 latest/best 写入 `weight/training/`；模型包发布时不携带这些本地训练权重或 `result/` 生成物。

### 真实数据

真实训练需要 MRMS 瞬时/累计降水、OMO/ASOS 站点观测、HRRR 617 通道和 stale-age、GOES 16 通道、高程、统一投影、QC、缺测及归一化统计量。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本目录为根据 MetNet-3 论文构建的独立 compact 复现。
- 论文材料、代码和后续真实数据应分别遵循各自许可证。
