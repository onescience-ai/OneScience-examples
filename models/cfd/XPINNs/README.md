
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">XPINN</span>
  </strong>
</p>

# 模型介绍

XPINN（Extended Physics-Informed Neural Network）使用域分解方法将复杂计算区域拆分为多个子域，并为每个子域配置独立的神经网络。训练时除偏微分方程残差和外边界条件外，还会约束子域接口处的解连续性及残差一致性。

本模型包复现 XPINN 论文中的二维 Poisson 基准案例。X 形不规则区域被拆分为三个子域，分别使用 `tanh`、`sin` 和 `cos` 激活函数：

```text
Subdomain 1
|-- Interface 1 -- Subdomain 2
`-- Interface 2 -- Subdomain 3
```

控制方程：

```text
u_xx + u_yy = exp(x) + exp(y)
```

论文：Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations  
https://doi.org/10.4208/cicp.OA-2020-0164

# 仓库说明

本仓库是 OneScience 整理的 XPINN 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用统一 YAML 配置训练三子域 XPINN
- 分别配置三个子网络的层宽和激活函数
- 对各子域成对采样空间坐标
- 约束边界、PDE 残差、接口值和接口残差
- 计算整体及各子域相对 L2 误差
- 生成不规则区域预测和误差图
- 在 CPU、GPU 或 DCU 上运行

当前不支持能力：

- 不提供论文中的时空分解和其他非线性方程案例
- 不内置预训练权重，需先运行训练脚本生成检查点
- 默认训练步数用于示例验证，不代表论文报告的最终精度

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 复杂区域 PDE 求解 | 在 X 形不规则区域上求解二维 Poisson 方程 |
| 域分解研究 | 为不同子域配置独立网络结构和激活函数 |
| 接口约束研究 | 比较接口解连续性与残差一致性损失 |
| 模型流程验证 | 缩小真实 MATLAB 数据采样量检查训练和推理流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 数据、模型、损失及训练配置 | 路径相对于模型包根目录解析 |
| `model/xpinn.py` | 三子域 XPINN 模型及 Poisson 残差 | 基于 PyTorch 自动微分 |
| `scripts/train.py` | 训练脚本 | 保存带网络结构元数据的检查点 |
| `scripts/inference.py` | 推理、评估与可视化脚本 | 读取 `weight/xpinn_poisson_2d.pt` |
| `scripts/data_utils.py` | MATLAB 数据校验、采样及张量构造 | 保证坐标成对采样 |
| `data/XPINN_2D_PoissonEqn.mat` | XPINN 二维 Poisson 数据 | 包含三个子域、边界和接口数据 |
| `weight/` | 模型权重目录 | 训练前为空 |
| `result/` | 推理结果目录 | 首次推理时自动创建 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 完成正式训练。
- CPU 可用于小配置流程验证，完整采样训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/XPINNs --local_dir ./xpinn
cd xpinn
```

### 安装运行环境

已有 OneScience 环境时直接激活对应环境。新环境可安装最小依赖：

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy scipy matplotlib pyyaml
```

### 训练

使用 `conf/config.yaml` 中的默认采样数量训练：

```bash
python scripts/train.py
```

训练完成后，检查点默认保存至：

```text
weight/xpinn_poisson_2d.pt
```

可使用随包真实 MATLAB 数据做小配置连通性验证。请临时修改 `conf/config.yaml` 中的 `training.steps` 和 `data.samples`，例如：

```yaml
root:
  data:
    samples:
      residual_1: 8
      residual_2: 8
      residual_3: 8
      boundary: 8
      interface_1: 4
      interface_2: 4
  training:
    steps: 1
```

训练脚本完全由 `conf/config.yaml` 驱动，不接收命令行配置参数。

### 推理、评估和可视化

完成训练后执行：

```bash
python scripts/inference.py
```

推理脚本会输出整体相对 L2 误差，并生成：

```text
result/
  xpinn_poisson_2d.png
```

推理脚本同样只读取 `conf/config.yaml`，如需切换数据、权重或输出目录，请直接修改 YAML 文件。

# 配置说明

`conf/config.yaml` 的 `root.common` 控制设备、精度、随机种子和输出目录。其余配置分为：

- `data`：MATLAB 数据路径和各类训练样本数量
- `model`：三个子网络的层宽及激活函数
- `loss`：边界、PDE、接口残差和接口值损失权重
- `training`：Adam 步数、学习率、日志间隔和检查点名称
- `inference`：结果图文件名

`common.device` 设为 `auto` 时会优先使用 PyTorch 可见的 GPU 或 DCU，否则回退到 CPU。

# 数据格式

`data/XPINN_2D_PoissonEqn.mat` 的主要字段如下：

| 字段 | 说明 |
| --- | --- |
| `x_f1`、`y_f1` | 子域 1 内部点 |
| `x_f2`、`y_f2` | 子域 2 内部点 |
| `x_f3`、`y_f3` | 子域 3 内部点 |
| `xb`、`yb`、`ub` | 外边界坐标和边界值 |
| `xi1`、`yi1` | 子域 1 与子域 2 的接口点 |
| `xi2`、`yi2` | 子域 1 与子域 3 的接口点 |
| `u_exact1`、`u_exact2`、`u_exact3` | 各子域精确解 |
| `u_exact` | 按子域顺序拼接的整体精确解 |

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Jagtap, A. D., Kharazmi, E., and Karniadakis, G. E. Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations. Communications in Computational Physics, 28(5), 2002-2041, 2020.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
