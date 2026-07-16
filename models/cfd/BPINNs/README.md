
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">BPINNs</span>
  </strong>
</p>

# 模型介绍

BPINNs（Bayesian Physics-Informed Neural Networks）将物理方程约束引入贝叶斯神经网络，用于在数据存在噪声时同时估计方程解和预测不确定性。本模型包提供一维 Laplace 方程案例：

```text
u_xx + pi^2 sin(pi x) = 0,  x in [0, 1]
u(x) = sin(pi x)
```

整理后的默认训练流程保留原实现的 Adam 和 L-BFGS 优化，并支持从包含多个参数状态的检查点计算后验预测均值与标准差。默认训练只生成一个优化状态，因此其标准差为零；如需完整贝叶斯不确定性，需要额外提供 HMC、变分推断或集成采样得到的 `posterior_states`。

论文：B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data  
https://doi.org/10.1016/j.jcp.2020.109913

# 仓库说明

本仓库是 OneScience 整理的 BPINNs 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用统一 YAML 配置训练一维 Laplace 方程
- 组合观测数据、边界条件和 PDE 残差损失
- 使用 Adam 训练并通过 L-BFGS 自动精调
- 从已有检查点继续执行 L-BFGS 精调
- 兼容带元数据检查点和原始 PyTorch `state_dict`
- 从多个参数状态计算后验均值和标准差
- 输出相对 L2、RMSE、最大绝对误差及结果图
- 在 CPU、GPU 或 DCU 上运行

当前不支持能力：

- 默认训练入口不执行 HMC 或变分贝叶斯采样
- 不内置预训练权重，需先运行训练脚本生成检查点
- 不提供多维正问题和逆问题案例

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 一维 Laplace 求解 | 使用观测点、边界点和配点共同训练方程解 |
| 优化器对比 | 比较 Adam 与 L-BFGS 对 PINN 收敛的影响 |
| 检查点精调 | 从已有权重继续运行 L-BFGS |
| 后验预测 | 从包含多个参数状态的检查点计算预测均值和标准差 |
| 模型流程验证 | 缩小解析数据规模检查训练、精调和推理流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 模型、训练、损失和解析数据配置 | 路径相对于模型包根目录解析 |
| `model/bpinn.py` | BPINN 网络、Laplace 损失和后验预测 | 基于 PyTorch 自动微分 |
| `scripts/train.py` | Adam 和 L-BFGS 训练脚本 | 保存基础检查点和训练历史 |
| `scripts/refine.py` | L-BFGS 精调脚本 | 默认生成 refined 检查点 |
| `scripts/inference.py` | 推理、评估与可视化脚本 | 优先加载 refined 检查点 |
| `scripts/common.py` | 配置、解析数据、设备和指标工具 | 供三个入口共享 |
| `weight/` | 模型权重目录 | 训练前为空 |
| `data/` | 数据目录 | 当前解析方程案例不需要外部数据 |
| `result/` | 训练历史和推理结果目录 | 首次运行时自动创建 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 完成正式训练。
- CPU 可用于小配置流程验证。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/BPINNs --local_dir ./BPINNs
cd BPINNs
```

### 安装运行环境

已有 OneScience 环境时直接激活对应环境。新环境可安装最小依赖：

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy matplotlib pyyaml
```

### 训练

使用 `conf/config.yaml` 中的默认配置训练：

```bash
python scripts/train.py
```

训练完成后默认生成：

```text
weight/bpinn_laplace1d.pt
result/training_history.npz
```

训练脚本完全由 `conf/config.yaml` 驱动，不接收命令行配置参数。如需使用小配置完成流程验证，可临时修改以下字段：

```yaml
root:
  data:
    n_sol: 8
    n_pde: 8
    test_res: 50
  training:
    epochs: 1
    lbfgs_iters: 0
```

### L-BFGS 精调

对默认基础检查点继续精调：

```bash
python scripts/refine.py
```

精调后默认生成：

```text
weight/bpinn_laplace1d_refined.pt
```

精调迭代次数、学习率及输入输出检查点名称分别由 `training.lbfgs_iters`、`training.refinement_lr`、`training.checkpoint_name` 和 `training.refined_checkpoint_name` 控制。

### 推理、评估和可视化

```bash
python scripts/inference.py
```

存在 refined 检查点时会优先加载，否则加载基础检查点。推理结果默认保存为：

```text
result/
  bpinn_laplace1d.npz
  bpinn_laplace1d.png
```

推理脚本同样只读取 `conf/config.yaml`，不接收命令行配置参数。

# 配置说明

`conf/config.yaml` 的 `root.common` 控制设备、精度、随机种子和输出目录。其余配置分为：

- `model`：输入维度、隐藏层数量、宽度、输出和激活函数
- `training`：Adam 轮数、学习率、L-BFGS 轮数、精调学习率和检查点名称
- `loss`：观测数据、PDE 和边界损失权重
- `data`：计算域、观测点、配点、噪声和测试分辨率
- `inference`：预测、结果图和训练历史文件名

`common.device` 设为 `auto` 时会优先使用 PyTorch 可见的 GPU 或 DCU，否则回退到 CPU。

训练、精调和推理入口均固定读取模型包内的 `conf/config.yaml`。如需切换设备、输出目录、数据规模或优化器设置，请直接修改该 YAML 文件。

# 数据格式

当前案例使用解析函数生成数据，不依赖外部文件，因此 `data/` 默认保持为空。

训练数据包括：

| 数据 | 说明 |
| --- | --- |
| `x_solution`、`u_solution` | 域内随机观测点和解析解，可添加高斯噪声 |
| `x_boundary`、`u_boundary` | 区间两端的 Dirichlet 边界值 |
| `x_pde` | 域内 PDE 配点 |
| `x_test`、`u_test` | 等距评估网格和解析解 |

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Yang, L., Meng, X., and Karniadakis, G. E. B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data. Journal of Computational Physics, 425, 109913, 2021.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文说明。
