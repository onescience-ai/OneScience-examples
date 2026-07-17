<p align="center">
  <strong>
    <span style="font-size: 30px;">GP_for_TO</span>
  </strong>
</p>

# 模型介绍

GP_for_TO（Physics-informed GP-TO）是用于同步无网格拓扑优化的物理信息高斯过程方法。该方法使用共享神经网络均值函数和多个高斯过程输出联合表示速度 `u`、速度 `v`、压力 `p` 和材料密度 `ro`，并通过 PDE 残差、耗散功率和体积约束共同优化二维流体拓扑设计。   

# 仓库说明

本仓库是 OneScience 整理后的 GP_for_TO 标准运行包，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 生成最小假数据用于流程连通性验证
- 训练 GP_for_TO 拓扑优化模型
- 从 checkpoint 推理 `x/u/v/p/ro` 场变量
- 汇总推理结果并可选生成二维场图

当前不支持能力：

- 不内置预训练权重
- 不自动下载外部数据集
- 不提供分布式训练入口

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | OneCode 元信息 | 最小配置 |
| `conf/config.yaml` | 问题、模型、训练、推理和假数据配置 | 已适配本仓库相对路径 |
| `model/` | GPPLUS 相关模型文件 | 从 `onescience.models.GPs` 复制必要文件，本地相对导入 |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成 `data/fake/<problem>_samples.npz` 和 metadata |
| `scripts/train.py` | 训练脚本 | 默认读取 `conf/config.yaml`，支持命令行覆盖 |
| `scripts/inference.py` | 推理脚本 | 默认读取 `weight/gp_for_to.pt` |
| `scripts/result.py` | 推理结果摘要脚本 | 读取 `result/inference/predictions.npz` |
| `scripts/topology_optimization.py` | 拓扑优化损失和训练循环 | 由 `scripts/train.py` 调用 |
| `weight/` | 权重目录 | 默认 checkpoint 为 `weight/gp_for_to.pt` |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

# 快速开始

### 安装运行环境

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```


## 生成假数据
如需先验证脚本、模型、checkpoint 和结果文件是否能够完整跑通，可使用仓库内置脚本生成最小数据

```bash
python scripts/fake_data.py
```

默认会按 `conf/config.yaml` 生成 `doublepipe` 的运行时样本，输出到：

```text
data/fake/doublepipe_samples.npz
data/fake/doublepipe_metadata.json
```

## 训练

默认训练配置保留原始设置：`N_col_domain=10000`、`N_train_per_BC=25`、`num_iter=50000`、`diff_method=Numerical`。

```bash
python scripts/train.py
```

训练输出：

```text
weight/gp_for_to.pt
result/training/loss_history.npy
result/training/training_summary.json
```

如需低成本检查流程，可运行最小 smoke test：

```bash
python scripts/train.py --device cpu --num-iter 0 --n-col-domain 16 --n-train-per-bc 25 --diff-method Autograd --no-plot
```

## 推理

```bash
python scripts/inference.py
```

默认读取 `weight/gp_for_to.pt`，输出：

```text
result/inference/predictions.npz
result/inference/inference_summary.json
```

`predictions.npz` 中包含 `x`、`u`、`v`、`p`、`ro` 五组数组。

## 预测结果可视化

```bash
python scripts/result.py
```

该脚本会打印预测数组的 shape、dtype、最小值、最大值和均值，并默认生成：

```text
result/inference/field_summary.png
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- GP_for_TO 原始论文：[Simultaneous and Meshfree Topology Optimization with Physics-informed Gaussian Processes](https://arxiv.org/abs/2408.03490)。
- 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
