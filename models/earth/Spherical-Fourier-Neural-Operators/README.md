<p align="center">
  <strong><span style="font-size: 30px;">Spherical Fourier Neural Operator</span></strong>
</p>

# 模型介绍

SFNO 使用球谐变换在球面上学习动力系统演化，可用于全球天气预报和球面浅水方程预测。

论文：Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere

https://proceedings.mlr.press/v202/bonev23a.html

# 模型描述

当前模型包调用 NVIDIA 官方 `torch-harmonics` 线性 SFNO 实现，支持 fake 球面场上的 SHT、一次参数更新、checkpoint 恢复和短时自回归。它是算子级 smoke package，不是论文 SWE/ERA5 实验复现。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 球面算子研究 | 验证 SHT、频谱滤波和逆 SHT。 |
| 本地快速验证 | 使用 fake 球面数据跑通训练和推理。 |
| ERA5 天气预报 | 后续可接入 26 或 73 通道 ERA5 数据。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `download.sh` | ModelScope 资源下载脚本 | 当前无额外模型或数据文件需要下载 |
| `requirements.lock` | 第三方依赖锁定 | 固定 `torch-harmonics==0.8.0` |
| `THIRD_PARTY.md` | 第三方资源与许可证说明 | 记录上游版本、提交和本地安装方式 |
| `.deps/` | 项目内置依赖目录 | 保存可复现安装的 `torch-harmonics` 包及元数据 |
| `model/default_config.json` | 默认模型配置 | 定义球面网格、通道、层数和 rollout 参数 |
| `model/sfno_adapter.py` | 官方 SFNO 适配层 | 封装 `torch-harmonics` 的线性 SFNO 实现 |
| `model/dataset.py` | 相邻时间帧数据集 | 将球面序列组织为 input/target pair |
| `model/fake_spherical_data.py` | 合成球面场生成模块 | 生成确定性的低阶平滑测试序列 |
| `scripts/train.py` | 训练入口 | 执行训练、验证、early stopping 和 checkpoint 保存 |
| `scripts/inference.py` | 自回归推理入口 | 加载 checkpoint 并执行短时 rollout |
| `scripts/result.py` | 评估与可视化入口 | 输出 RMSE、空间 ACC 和场对比图 |
| `weight/` | 本地训练权重目录 | 保存训练产生的 checkpoint |

# 使用说明

## 1. OneCode 使用

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- CPU 可运行当前小配置。
- 完整 ERA5 训练推荐使用 GPU。


## 3. 快速开始
### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
pip install torch-harmonics==0.8.0
```

**GPU 环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
pip install torch-harmonics==0.8.0
```

当前目录也在 `.deps/` 中保留了 `torch-harmonics==0.8.0`。

### 数据

当前脚本在内存中生成低阶平滑 fake 球面场，并将相邻时间帧切成 `T-1` 个 input/target pair；不需要额外下载数据。

### 训练

```bash
python scripts/train.py
```

训练现在执行 pair Dataset 的多 epoch 训练、按时间顺序划分验证集、学习率调度和 early stopping：

```bash
python scripts/train.py --epochs 10
python scripts/train.py --resume weight/training/latest.pth --epochs 20
```

### 推理

```bash
python scripts/inference.py
```

输出文件：

```text
weight/model.pth
weight/training/latest.pth
weight/training/best.pth
weight/training/history.json
result/prediction.pt
result/target.pt
result/inference.json
```

### 结果检验

```bash
python scripts/result.py
```

当前测试只验证模型可运行。随机初始化 rollout 不代表论文长期稳定性结果。

结果脚本生成 `result/metrics.json` 和 `result/comparison.png`。当前 RMSE 未做球面积分权重，ACC 使用样本自身空间均值而非训练集长期气候态，因此不能与论文指标比较。

### 论文与当前实现的 I/O

| 项目 | 论文 SWE / ERA5 | 当前 smoke 配置 |
| --- | --- | --- |
| 输入输出 | SWE 3 场 `256x512` / ERA5 26 或 73 通道 | `[B,2,17,32]` 平滑合成场 |
| 时间步长 | SWE 1 小时 / ERA5 6 小时 | 无物理单位的相邻序号 |
| 架构 | SWE 4x256；天气模型 8x384 | 2 blocks，embed dim 8 |
| 训练 | 单步训练后双步自回归微调 | 多 epoch 单步 pair 训练和验证；rollout 用于推理分析 |
| 分析 | 球面加权相对误差和气候态 ACC | 未加权 smoke RMSE/ACC |

完整执行顺序为 `train.py -> inference.py -> result.py`。训练在内存中生成 z-score 后的 `[T,C,Nlat,Nlon]`，并按相邻帧构成 pair。checkpoint 的 `config` 只保存模型配置，训练参数单独放在 `train_config`，因此推理可以继续从 checkpoint 恢复模型结构。模型包发布时不携带本地训练权重或 `result/` 生成物。真实 SWE/ERA5 模式仍需数据读取、变量表、正式划分、面积加权损失和论文的两阶段训练循环。

### 真实数据

真实训练需要准备 ERA5 26/73 通道数据、6 小时时间配对、训练集统计量和球面网格重采样配置。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 官方实现：https://github.com/NVIDIA/torch-harmonics
- 本目录为 SFNO 的独立运行适配，第三方条款见 `THIRD_PARTY.md`。
