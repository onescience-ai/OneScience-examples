<p align="center"><strong><span style="font-size: 30px;">PINN-CFD-Multiphysics-2606-21945</span></strong></p>

# 模型介绍

这是一个基于物理信息神经网络（PINN）的计算流体力学复现实验包。模型以 `(x, y, t)` 坐标为输入，预测三个原始场变量，并通过 Navier-Stokes 残差、边界条件和初始条件构造训练目标。该产物来自论文复现任务 `2606.21945` 的 Tier 1 可执行代表性实现；原综述论文没有规定统一 CFD 数据集、几何、基准求解器或数值目标。

# 模型描述

- `model/models/pinn_mlp.py`: `[3, 16, 16, 3]` 的可配置全连接 PINN MLP，默认使用 `tanh`。
- `model/models/residuals.py`: Navier-Stokes 物理残差计算。
- `model/losses/pinn_loss.py`: physics、BC、IC 和可选 data 项的组合损失。
- 推理为单次连续 `(x, y, t)` 查询，不执行自回归 rollout。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用 `scripts/train.py` 和 `conf/tier1_cfd.yaml` 从头训练。 |
| 模型推理 | 使用 `scripts/infer.py` 加载 `weight/best.pt` 对坐标张量预测。 |
| 评估和可视化 | 使用 `scripts/evaluate.py` 计算 PDE residual MSE 与 continuity MSE。 |
| 本地验证 | CPU smoke-sized sampling；完整基准比较仍缺少统一参考场。 |

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

```bash
modelscope download --model OneScience/PINN-CFD-Multiphysics-2606-21945 --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**
```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本 Tier 1 实现不依赖外部观测数据集，使用矩形域内的随机配点和零 BC/IC 工程假设。数据域、采样数量和物理黏性参数位于 `conf/tier1_cfd.yaml`。

### 训练

```bash
python scripts/train.py --config conf/tier1_cfd.yaml --output-dir results
```

### 训练权重

已提供 `weight/best.pt`，该检查点包含 `model`、`model_kwargs`、训练配置和验证指标。

### 推理

```bash
python scripts/infer.py --checkpoint weight/best.pt --coordinates coordinates.pt --output predictions.pt
```

其中 `coordinates.pt` 必须是形状为 `[N, 3]` 的 CPU 张量。

### 评估和可视化

```bash
python scripts/evaluate.py --checkpoint weight/best.pt --coordinates coordinates.pt --output evaluation.json
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本包为论文复现任务 `2606.21945` 的 Tier 1 代表性实现。
- 原综述论文未定义作者专属 CFD 数据集、统一参考解或数值基准；请在正式发布时补充准确论文引用信息。
- 代码和发布模板采用 Apache License 2.0；请确认上游论文与依赖的许可条款。

