
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">FPINNs</span>
  </strong>
</p>

# 模型介绍

本模型包中的 FPINNs 指 Fuzzy Physics-Informed Neural Networks，而不是分数阶 PINNs。模型在标准全连接网络旁增加高斯模糊隶属度分支，将神经特征和模糊规则特征拼接后预测偏微分方程解。

```text
输入 (x, t)
  |-- FCN trunk ------------------|
  `-- Gaussian fuzzy rules -------|-- feature fusion -- u(x, t)
```

当前案例求解 Allen-Cahn 方程：

```text
u_t - lambda_1 u_xx + lambda_2 (u^3 - u) = 0
```

支持两个任务：

- `forward`：已知 `lambda_1=0.0001`、`lambda_2=5.0`，预测 `u(x,t)`
- `inverse`：根据观测数据同时学习 `u(x,t)`、`lambda_1` 和 `lambda_2`

论文：Deep fuzzy physics-informed neural networks for forward and inverse PDE problems  
https://doi.org/10.1016/j.neunet.2024.106750

# 仓库说明

本仓库是 OneScience 整理的 FPINNs 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用统一 YAML 配置管理正向和逆向任务
- 使用 FCN 与高斯模糊规则双分支网络
- 从 Allen-Cahn MATLAB 数据随机采样训练点
- 使用数据损失和 PDE 残差联合训练
- 使用 Adam 训练并可选执行 L-BFGS 精调
- 逆向任务同步学习两个方程参数
- 分批评估完整 `512×201` 时空网格
- 输出相对 L2、RMSE、最大绝对误差和场图
- 在 CPU、GPU 或 DCU 上运行

当前不支持能力：

- 不提供论文中的其他 PDE 案例
- 不内置预训练权重，需先运行训练脚本生成检查点
- 默认逆向参数未施加正值参数化约束，与原实现保持一致

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| Allen-Cahn 正向求解 | 使用已知扩散和反应参数预测完整时空解 |
| Allen-Cahn 逆向求解 | 从解观测中识别扩散和反应参数 |
| 模糊特征研究 | 比较 FCN 特征和高斯模糊规则融合效果 |
| 模型流程验证 | 缩小真实 MATLAB 数据采样量检查训练和推理流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 正向、逆向、模型、数据及输出配置 | 路径相对于模型包根目录解析 |
| `model/fpinn.py` | FCN、FuzzyLayer、FPINN 和 Allen-Cahn 损失 | 基于 PyTorch 自动微分 |
| `scripts/train.py` | 统一训练脚本 | 通过 YAML 选择正向或逆向任务 |
| `scripts/inference.py` | 推理、评估与时空场可视化 | 分批处理完整测试网格 |
| `scripts/data_utils.py` | MATLAB 数据校验、采样和分批预测 | 保持 `[t,x]` 场顺序 |
| `scripts/common.py` | 配置、设备、随机种子和检查点工具 | 供训练和推理共享 |
| `data/AC.mat` | Allen-Cahn 数据 | 包含 `x`、`tt`、`uu` |
| `weight/` | 模型权重目录 | 训练前为空 |
| `result/` | 训练历史和推理结果目录 | 首次运行时自动创建 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 完成正式训练和完整网格推理。
- CPU 可用于小配置流程验证。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/FPINNs --local_dir ./FPINNs
cd FPINNs
```

### 安装运行环境

已有 OneScience 环境时直接激活对应环境。新环境可安装最小依赖：

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy scipy matplotlib pyyaml
```

### 训练

训练任务由 `conf/config.yaml` 中的 `common.task` 控制，可设置为 `forward` 或 `inverse`。配置完成后运行：

```bash
python scripts/train.py
```

检查点和训练历史分别保存至 `weight/` 和 `result/`。

可使用随包真实数据做小配置流程验证，例如验证正向任务：

```yaml
root:
  common:
    task: "forward"
  data:
    n_train: 8
  tasks:
    forward:
      training:
        epochs: 1
        lbfgs_iters: 0
```

启用 L-BFGS 精调时，直接设置当前任务下的 `training.lbfgs_iters`。训练脚本完全由 `conf/config.yaml` 驱动，不接收命令行配置参数。

### 推理、评估和可视化

完成对应任务训练后执行：

```bash
python scripts/inference.py
```

推理任务同样由 `common.task` 控制。推理脚本只读取 `conf/config.yaml`，不接收命令行配置参数。

默认生成：

```text
result/
  fpinn_forward.npz
  fpinn_forward.png
  fpinn_inverse.npz
  fpinn_inverse.png
```

逆向推理还会在终端和 NPZ 文件中输出恢复得到的 `lambda_1` 和 `lambda_2`。

# 配置说明

`conf/config.yaml` 的 `root.common` 控制任务、设备、精度、随机种子和输出目录。其余配置分为：

- `data`：MATLAB 文件、训练采样点数量和评估批大小
- `model`：FCN 隐藏层、神经特征维度、模糊规则数和激活函数
- `tasks.forward`：正向训练参数、损失权重和已知 PDE 参数
- `tasks.inverse`：逆向训练参数、损失权重、参数初值和真实值
- `output`：检查点、训练历史、预测和结果图文件名

`common.device` 设为 `auto` 时会优先使用 PyTorch 可见的 GPU 或 DCU，否则回退到 CPU。

`common.task` 支持 `forward` 和 `inverse`。训练和推理入口均固定读取模型包内的 `conf/config.yaml`；如需切换任务、设备、数据、输出目录或训练规模，请直接修改该 YAML 文件。

# 数据格式

`data/AC.mat` 包含：

| 字段 | 原始形状 | 说明 |
| --- | --- | --- |
| `x` | `[1, 512]` | 空间坐标 |
| `tt` | `[1, 201]` | 时间坐标 |
| `uu` | `[512, 201]` | Allen-Cahn 解，轴顺序为 `[x,t]` |

加载后会构造：

- 坐标 `[x,t]`，形状为 `[102912, 2]`
- 解向量，形状为 `[102912, 1]`
- 可视化场，形状为 `[201, 512]`，轴顺序为 `[t,x]`

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Wu, W., Duan, S., Sun, Y., Yu, Y., Liu, D., and Peng, D. Deep fuzzy physics-informed neural networks for forward and inverse PDE problems. Neural Networks, 181, 106750, 2025.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
