
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">SA-PINNs</span>
  </strong>
</p>

# 模型介绍

SA-PINNs（Self-Adaptive Physics-Informed Neural Networks）为每个观测点、边界点和 PDE 配点引入可学习的正注意力权重。训练采用 min-max 鞍点优化：网络参数通过梯度下降减小加权物理损失，注意力参数通过梯度上升提高难拟合位置的权重。

注意力权重定义为归一化的 `exp(alpha)`，训练过程中保持正值且均值接近 1。Adam 阶段同时更新网络和注意力，L-BFGS 阶段冻结注意力并只精调网络。

本模型包提供三个案例：

- 一维 Laplace 方程
- 二维 Helmholtz 方程
- 一维时变 Burgers 方程

论文：Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism  
https://arxiv.org/abs/2009.04544

# 仓库说明

本仓库是 OneScience 整理的 SA-PINNs 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用统一 YAML 配置管理三个 PDE 案例
- 为数据、边界和 PDE 配点分别学习注意力权重
- 通过梯度下降/上升执行 min-max Adam 优化
- 冻结注意力后使用 L-BFGS 精调网络
- 保存模型、注意力参数、采样配置和训练历史
- 对 Laplace 和 Helmholtz 计算相对 L2 误差
- 绘制预测、误差及 PDE 注意力分布
- 在 CPU、GPU 或 DCU 上运行

当前不支持能力：

- Burgers 案例不内置参考解，因此只输出预测场和注意力分布
- 不提供原目录中仅有占位声明、未实现方程和数据的 Allen-Cahn 案例
- 不内置预训练权重，需先运行训练脚本生成检查点

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 一维 Laplace 求解 | 观察注意力如何聚焦高残差配点 |
| 二维 Helmholtz 求解 | 处理具有高频空间变化的二维解析解 |
| Burgers 方程求解 | 联合初值、边界和非线性 PDE 残差训练 |
| 自适应权重研究 | 比较普通 PINN 和逐点 soft-attention 加权 |
| 模型流程验证 | 缩小解析采样规模检查训练和推理流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 三案例模型、训练、损失及数据配置 | 路径相对于模型包根目录解析 |
| `model/sa_pinn.py` | FCN、注意力、SA-PINN、方程和加权损失 | 基于 PyTorch 自动微分 |
| `scripts/train.py` | 统一训练脚本 | 支持 `laplace`、`helmholtz`、`burgers` |
| `scripts/inference.py` | 推理、评估与可视化脚本 | 按案例读取对应权重 |
| `scripts/problems.py` | 三个方程的解析解和数据构造 | Burgers 使用初边值约束 |
| `scripts/common.py` | 配置、设备、随机种子和检查点工具 | 供训练和推理共享 |
| `weight/` | 模型权重目录 | 训练前为空 |
| `data/` | 数据目录 | 当前解析案例不需要外部数据 |
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
modelscope download --model OneScience/SA-PINNs --local_dir ./SA-PINNs
cd SA-PINNs
```

### 安装运行环境

已有 OneScience 环境时直接激活对应环境。新环境可安装最小依赖：

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy matplotlib pyyaml
```

### 训练

分别训练三个案例：

```bash
python scripts/train.py --case laplace
python scripts/train.py --case helmholtz
python scripts/train.py --case burgers
```

检查点和训练历史分别保存至 `weight/` 和 `result/`。

可使用小配置完成真实解析数据上的流程验证：

```bash
python scripts/train.py --case laplace \
  --epochs 1 --lbfgs-iters 0 \
  --n-sol 8 --n-pde 8 --n-bnd 2 --test-res 20

python scripts/train.py --case helmholtz \
  --epochs 1 --lbfgs-iters 0 \
  --n-sol 8 --n-pde 8 --n-bnd 8 --test-res 20

python scripts/train.py --case burgers \
  --epochs 1 --lbfgs-iters 0 \
  --n-sol 8 --n-pde 8 --n-bnd 8 --test-res 20
```

命令行参数优先于 `conf/config.yaml`，例如 `--device cpu`、`--weight-dir /tmp/sapinn_weight` 和 `--result-dir /tmp/sapinn_result`。

### 推理、评估和可视化

训练对应案例后执行：

```bash
python scripts/inference.py --case laplace
python scripts/inference.py --case helmholtz
python scripts/inference.py --case burgers
```

默认生成：

```text
result/
  sapinn_laplace.npz
  sapinn_laplace.png
  sapinn_helmholtz.npz
  sapinn_helmholtz.png
  sapinn_burgers.npz
  sapinn_burgers.png
```

# 配置说明

`conf/config.yaml` 的 `root.common` 控制设备、精度、随机种子和输出目录。每个案例位于 `root.cases.<case>`，并包含：

- `model`：输入维度、隐藏层数量、宽度和激活函数
- `attention`：是否启用逐点注意力及其梯度上升学习率
- `training`：Adam 轮数、网络学习率、L-BFGS 轮数和日志间隔
- `loss`：数据、PDE 和边界损失权重
- `data`：计算域、观测点、配点、边界点及测试分辨率
- `output`：检查点、预测、结果图和训练历史文件名

`common.device` 设为 `auto` 时会优先使用 PyTorch 可见的 GPU 或 DCU，否则回退到 CPU。

# 数据格式

三个案例均使用解析方程和随机采样构造训练数据，不依赖外部文件，因此 `data/` 默认保持为空。

| 数据 | 说明 |
| --- | --- |
| `x_data`、`u_data` | Laplace/Helmholtz 观测点或 Burgers 初值点 |
| `x_boundary`、`u_boundary` | Dirichlet 边界点和边界值 |
| `x_pde` | 方程域内 PDE 配点 |
| `x_test`、`u_exact` | 推理网格和解析解；Burgers 的解析解为空 |

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- McClenny, L. and Braga-Neto, U. Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism. arXiv:2009.04544, 2020.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文说明。
