<p align="center">
  <strong>
    <span style="font-size: 30px;">CFD-HomogenizationTank-Forecaster</span>
  </strong>
</p>

# 模型介绍

本模型是论文《A Study on the Performance of Distributed Training of Data-driven CFD Simulations》（arXiv:2604.27431）的复现实现。它基于循环神经网络（RNN/LSTM）对均化水箱（homogenization tank）的 3D 流速场进行时序预测：给定最近 3 个 timestep 的扁平化速度场，预测下一个 timestep 的完整速度场。论文同时比较了仅 CPU、多 GPU 与分布式（Horovod / TensorFlow MultiWorker）训练策略的性能；本实现采用 PyTorch 重写，提供单卡与分布式（torch DDP + RCCL）训练路径，并复现了模型的精度指标。

论文：S. Iserte, A. González-Barberá, P. Barreda, K. Rojek, *A Study on the Performance of Distributed Training of Data-driven CFD Simulations*, Int. J. High Performance Computing Applications 37, 503–515 (2023)。
https://arxiv.org/abs/2604.27431

# 模型描述

- 输入：(batch, n_input=3, n_features)，其中 n_features = n_cells × 3（3D 速度 u/v/w over cells）
- 架构：LSTM(200) → RepeatVector → LSTM(200) → Dense(100, relu) → Dense(n_features, 线性)
- 损失：MAE；优化器：Adam(lr=0.00025)；batch_size=14
- 关键文件：`model/model.py`（RNN 模型）、`model/config.py`（配置）

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 时序预测训练 | 使用合成/CFD 流速场数据训练 RNN 模型（`scripts/train.py`） |
| 分布式训练 | 通过 SLURM + torch DDP + RCCL 多节点训练（`sbatch ddp.sbatch`） |
| 精度评估 | 计算 MAE / RMSE / Pearson / Spearman / R^2（`scripts/evaluate.py`） |
| 模型推理 | 加载权重对下一 timestep 流速场进行预测 |
| 数据生成 | 生成与论文数据结构一致的合成代理流速场（`scripts/make_dataset.py`） |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

# 根据当前仓库自行设置
```bash
modelscope download --model OneScience/CFD-HomogenizationTank-Forecaster --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem、bio、all（全领域）
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

论文真实 CFD 均化水箱数据集（131 案例 × 420 timesteps × 125,565 cells × 3，约 38.6GB）需联系作者单独获取（"provided on demand"）。本仓库使用一个与论文数据结构一致的**合成代理数据集**（共享固定几何基 + 单车例 inflow 驱动演化）：

```bash
python scripts/make_dataset.py --n-cases 131 --n-cells 1024 --n-steps 420 --outdir ../data/tier2
```

### 训练

单卡（DCU/CPU）：

```bash
python scripts/train.py --epochs 40 --batch-size 14 --lr 0.00025 \
  --n-cells 1024 --data-dir ../data/tier2 --save-dir ../output/tier2
```

多卡分布式（2 节点 × 1 DCU，RCCL）：

```bash
sbatch ddp.sbatch
# ddp.sbatch 内通过 SLURM 设置 RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT，调用：
# python train.py --distributed --backend nccl --data-dir ../data/tier2 --save-dir ../output/ddp_test
```

最佳模型保存于 `output/tier2/best_model.pt`。

### 训练权重

权重文件 `weight/best_model.pt`（PyTorch state dict，n_cells=1024，40 epochs）。

### 推理

加载 `weight/best_model.pt` 后对下一 timestep 流速场进行预测。推理入口见 `scripts/evaluate.py`（`--mode one-step` 或 `--mode recursive`）。

```bash
python scripts/evaluate.py --mode one-step --data-dir ../data/tier2 --save-dir ../output/tier2
```

### 评估和可视化

```bash
python scripts/evaluate.py --mode one-step --data-dir ../data/tier2 --save-dir ../output/tier2
```

评估指标：MAE / RMSE / Pearson / Spearman / R²。复现结果（n_cells=1024）：Pearson 0.968、Spearman 0.978、R² 0.913、训练 MAE ~0.03，与论文目标（0.99 / 0.98 / 0.98 / ~1e-2）趋势对齐。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为《A Study on the Performance of Distributed Training of Data-driven CFD Simulations》原始论文的复现版本。
- 官方代码上游：https://github.com/AlejandroGB13/CFD_AI
- License: Apache License 2.0（本仓库）；上游代码遵循其原始许可。