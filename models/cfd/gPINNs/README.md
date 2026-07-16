
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">gPINN</span>
  </strong>
</p>

# 模型介绍

gPINN（Gradient-enhanced Physics-Informed Neural Network）在传统 PINN 的偏微分方程残差约束之外，进一步约束残差对输入坐标的梯度。额外的高阶导数信息能够提高训练数据利用率，并改善部分正问题和反问题的求解精度。

本模型包提供三个经典方程案例：

- 一维 Poisson 方程
- 二维 Poisson 方程
- 带残差自适应加密（RAR）的 Burgers 方程

论文：Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems  
https://doi.org/10.1016/j.cma.2022.114823

# 仓库说明

本仓库是 OneScience 整理的 gPINN 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用统一 YAML 配置训练三个 PDE 案例
- 分案例或批量执行训练与推理
- 计算相对 L2 误差并生成结果图
- 对 Burgers 方程执行 RAR 自适应采样
- 在 CPU、GPU 或 DCU 上运行

当前不支持能力：

- 不提供论文全部反问题实验
- 随包检查点仅用于执行和格式验证，不代表论文报告的最终精度

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 一维 Poisson 求解 | 验证残差梯度约束和边界损失的组合训练流程 |
| 二维 Poisson 求解 | 使用硬边界输出变换求解零 Dirichlet 边界问题 |
| Burgers 方程求解 | 使用高阶自动微分和 RAR 加密高残差区域 |
| 模型流程验证 | 使用随包数据和检查点快速检查训练、推理及绘图流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 训练、推理、数据及输出配置 | 路径相对于模型包根目录解析 |
| `model/gpinn.py` | gPINN 网络、方程残差及梯度损失 | 基于 PyTorch 自动微分 |
| `scripts/train.py` | 统一训练脚本 | 支持 `1d`、`2d`、`burgers` 和 `all` |
| `scripts/inference.py` | 推理、评估与可视化脚本 | 读取 `weight/*.pt` |
| `data/Burgers.npz` | Burgers 方程参考解 | 包含 `t`、`x` 和 `usol` |
| `weight/` | 模型权重目录 | 默认保存和读取三个案例的检查点 |
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
modelscope download --model OneScience/gPINNs --local_dir ./gpinn
cd gpinn
```

### 安装运行环境

已有 OneScience 环境时直接激活对应环境。新环境可安装最小依赖：

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy matplotlib pyyaml
```

### 训练

训练全部案例：

```bash
python scripts/train.py --case all
```

也可以单独训练一个案例：

```bash
python scripts/train.py --case 1d
python scripts/train.py --case 2d
python scripts/train.py --case burgers
```

默认参数用于正式优化，训练时间较长。可使用随包真实数据做小配置连通性验证：

```bash
python scripts/train.py --case all --epochs 1 --nf 8 --lbfgs-iters 0 --quick
```

检查点默认保存至 `weight/`。命令行参数优先于 `conf/config.yaml`，例如 `--device cpu`、`--weight-dir /tmp/gpinn_weight` 和 `--data /path/to/Burgers.npz`。

### 推理、评估和可视化

对全部案例执行推理：

```bash
python scripts/inference.py --case all
```

也可以通过 `--case 1d`、`--case 2d` 或 `--case burgers` 仅运行一个案例。脚本会输出相对 L2 误差，并在 `result/` 下生成：

```text
result/
  gpinn_poisson1d.png
  gpinn_poisson2d.png
  gpinn_burgers.png
```

# 配置说明

`conf/config.yaml` 的 `root.common` 控制设备、精度、随机种子和输出目录。三个案例分别位于 `root.poisson1d`、`root.poisson2d` 和 `root.burgers`。

关键参数：

- `nf`：方程域内的配点数量
- `epochs`：Adam 优化迭代数
- `w_g`：残差梯度损失权重
- `layers`：全连接网络各层宽度
- `lbfgs_iters`：一维 Poisson 的 L-BFGS 优化迭代数
- `rar_rounds`：Burgers 方程的残差自适应加密轮数

`common.device` 设为 `auto` 时会优先使用 PyTorch 可见的 GPU 或 DCU，否则回退到 CPU。

# 数据格式

随包 `data/Burgers.npz` 包含以下数组：

| 数组 | 形状 | 说明 |
| --- | --- | --- |
| `t` | `[100, 1]` | 时间坐标 |
| `x` | `[256, 1]` | 空间坐标 |
| `usol` | `[256, 100]` | 参考解，轴顺序为 `[x, t]` |

一维和二维 Poisson 案例使用解析解构造源项及评估数据，不依赖外部数据文件。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Yu, J., Lu, L., Meng, X., and Karniadakis, G. E. Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems. Computer Methods in Applied Mechanics and Engineering, 393, 114823, 2022.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
