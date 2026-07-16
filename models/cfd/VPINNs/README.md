
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">hp-VPINN</span>
  </strong>
</p>

# 模型介绍

hp-VPINN（Variational Physics-Informed Neural Network with h- and p-Refinement）使用偏微分方程的变分弱形式训练神经网络。模型通过测试函数对方程残差进行积分约束，并结合域分解（h-refinement）和高阶多项式测试函数（p-refinement）提高复杂解的表达和求解能力。

本模型包提供两个经典方程案例：

- 一维 Poisson 方程
- 二维 Poisson 方程

论文：hp-VPINNs: Variational physics-informed neural networks with domain decomposition  
https://doi.org/10.1016/j.cma.2020.113547

# 仓库说明

本仓库是 OneScience 整理的 hp-VPINN 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用统一 YAML 配置训练一维和二维 Poisson 方程
- 分案例或批量执行训练与推理
- 使用 Gauss-Lobatto-Jacobi 数值积分构造变分损失
- 分别配置一维子域和二维非方形子域网格
- 计算相对 L2 误差并生成结果图
- 在 CPU、GPU 或 DCU 上运行

当前不支持能力：

- 不提供论文中的全部正问题和反问题实验
- 随包仅提供一维 Poisson 示例检查点；二维检查点需通过训练生成
- 随包检查点用于执行和格式验证，不代表论文报告的最终精度

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 一维 Poisson 求解 | 使用分段积分域和高阶测试函数约束弱形式残差 |
| 二维 Poisson 求解 | 在二维域分解网格上执行张量积数值积分 |
| h-refinement 研究 | 调整子域数量并比较空间分解效果 |
| p-refinement 研究 | 调整测试函数阶数并比较弱形式约束能力 |
| 模型流程验证 | 使用小规模积分配置检查训练、权重加载和绘图流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 训练、推理及输出配置 | 路径相对于模型包根目录解析 |
| `model/vpinn.py` | VPINN 网络、Jacobi 测试函数和数值积分 | 支持新旧检查点格式 |
| `scripts/train.py` | 统一训练脚本 | 支持 `1d`、`2d` 和 `all` |
| `scripts/inference.py` | 推理、评估与可视化脚本 | 读取 `weight/*.pt` |
| `weight/` | 模型权重目录 | 随包提供一维 Poisson 检查点 |
| `data/` | 数据目录 | 当前解析方程案例不需要外部数据 |
| `result/` | 推理结果目录 | 首次推理时自动创建 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 完成正式训练。
- CPU 可用于推理和小配置流程验证，完整训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/VPINNs --local_dir ./vpinn
cd vpinn
```

### 安装运行环境

已有 OneScience 环境时直接激活对应环境。新环境可安装最小依赖：

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy scipy matplotlib pyyaml
```

### 训练

训练一维 Poisson 案例：

```bash
python scripts/train.py
```

训练案例由 `conf/config.yaml` 中的 `common.case` 控制，可设置为 `1d`、`2d` 或 `all`。

默认参数用于正式优化，训练时间较长。可通过 YAML 使用小配置完成全部案例的训练流程验证：

```yaml
root:
  common:
    case: "all"
  poisson1d:
    n_element: 2
    n_test: 2
    n_quad: 4
    epochs: 1
    lbfgs_iters: 0
  poisson2d:
    n_el_x: 1
    n_el_y: 2
    n_test: 2
    n_quad: 4
    n_bound: 4
    epochs: 1
    lbfgs_iters: 0
```

检查点默认保存至 `weight/`。训练脚本完全由 `conf/config.yaml` 驱动，不接收命令行配置参数。

### 推理、评估和可视化

随包一维权重可直接执行：

```bash
python scripts/inference.py
```

训练二维权重后，将 `common.case` 设置为 `2d` 或 `all` 再运行同一命令。推理脚本同样只读取 `conf/config.yaml`，不接收命令行配置参数。

脚本会输出相对 L2 误差，并在 `result/` 下生成：

```text
result/
  hpvpinn_poisson1d.png
  hpvpinn_poisson2d.png
```

# 配置说明

`conf/config.yaml` 的 `root.common` 控制运行案例、设备、精度、随机种子和输出目录。两个案例分别位于 `root.poisson1d` 和 `root.poisson2d`。

关键参数：

- `n_element`：一维计算域的子域数量
- `case`：运行案例，支持 `1d`、`2d` 和 `all`
- `n_el_x`、`n_el_y`：二维计算域在两个方向上的子域数量
- `n_test`：每个方向的 Jacobi 测试函数数量
- `n_quad`：每个方向的 Gauss-Lobatto-Jacobi 积分点数量
- `epochs`：Adam 优化迭代数
- `lbfgs_iters`：L-BFGS 优化迭代数
- `layers`：全连接网络各层宽度
- `boundary_weight`：二维边界损失权重

`common.device` 设为 `auto` 时会优先使用 PyTorch 可见的 GPU 或 DCU，否则回退到 CPU。

训练和推理入口均固定读取模型包内的 `conf/config.yaml`。如需切换案例、设备、输出目录或训练规模，请直接修改该 YAML 文件。

# 数据格式

一维和二维 Poisson 案例均使用解析解构造源项、边界值及评估数据，不依赖外部数据文件，因此 `data/` 默认保持为空。

一维解析解：

```text
u(x) = 0.1 sin(8 pi x) + tanh(80 x)
```

二维解析解：

```text
u(x, y) = (0.1 sin(2 pi x) + tanh(10 x)) sin(2 pi y)
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Kharazmi, E., Zhang, Z., and Karniadakis, G. E. hp-VPINNs: Variational physics-informed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering, 374, 113547, 2021.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文说明。
