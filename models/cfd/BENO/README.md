---
license: apache-2.0
language:
- en
- zh
tags:
- OneScience
- BENO
- 椭圆型PDE
- 边界嵌入神经算子
frameworks: PyTorch
---

<p align="center">
  <strong>
    <span style="font-size: 30px;">BENO</span>
  </strong>
</p>

# 模型介绍

BENO 是北京大学、西湖大学和斯坦福大学提出的用于复杂边界条件下椭圆型 PDE 求解的边界嵌入神经算子。该模型面向 Poisson/Laplace 等稳态边值问题，通过双分支 GNN 分别建模内部源项和边界影响，并利用 Transformer 编码全局边界信息，从而更好地处理复杂边界几何和非齐次边界值。它适用于复杂边界 PDE 求解、稳态物理场预测、非齐次边界条件建模以及传统 FEM/FDM 求解器加速等场景。


# 仓库说明

本仓库是 OneScience 整理的 BENO 最小可运行标准工程，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

* 训练
* 推理
* 评估与可视化
* 生成空壳假数据用于流程连通性验证

当前不支持能力：

- 不内置预训练权重
- 不负责自动下载、清洗或重新适配全部外部数据库

# 适用场景

| 场景         | 说明                                |
| ---------- | --------------------------------- |
| 椭圆型 PDE 求解 | 面向 Poisson、Laplace 等稳态边值问题的快速近似求解 |
| 复杂边界条件建模   | 处理自由形状边界、不规则区域和非齐次边界值对解场的影响       |
| 稳态物理场预测    | 预测由源项和边界共同决定的平衡态物理场分布             |
| 数值求解器代理加速  | 替代或辅助 FEM、FDM、FVM 等传统求解流程，提高推理效率  |


# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | ModelScope/OneCode 元信息 | 最小配置 |
| `conf/config.yaml` | 数据、模型、训练和推理配置 | 默认 tiny smoke test 配置 |
| `model/` | BENO 图网络模型包 | OneScience复现的经典TOP模型 |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成 `RHS/SOL/BC_<file_prefix>_all.npy` |
| `scripts/train.py` | 训练脚本 | 默认 CPU smoke 配置，可改为 `cuda` 或 `auto` |
| `scripts/inference.py` | 推理脚本 | 自动读取 `weight/model_epoch_*.pt` 中最新权重 |
| `scripts/result.py` | 指标查看脚本 | 读取 `result/metrics.json` |
| `weight/` | 权重目录 | 默认保存 `model_epoch_<epoch>.pt` |


# 使用说明
## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/BENO --local_dir ./BENO
cd BENO
```

### 安装运行环境

```bash
git clone https://gitee.com/onescience-ai/onescience.git
cd onescience
bash install.sh cfd
```


### 生成假数据进行流程验证
直接生成最小数据检查训练和推理链路。
```bash
python scripts/fake_data.py
```

OneScience 社区提供可供训练的 `beno` 数据，用户可通过下述命令下载，并确认 `config/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/beno --local_dir ./data
```
完整的数据集文件也可通过[官方链接](https://drive.google.com/file/d/11PbUrzJ-b18VhFGY_uICSciCkeGrsaTZ/view)下载

### 训练

```bash
python scripts/train.py
```

默认 1 个 epoch，会保存：

```text
weight/model_epoch_0.pt
```
### 推理

```bash
python scripts/inference.py
```

### 评估和可视化

```bash
python scripts/result.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- BENO 原始论文：[BENO: Boundary-embedded Neural Operators for Elliptic PDEs](https://openreview.net/forum?id=ZZTkLDRmkg)
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理；公开分发前请根据上游项目确认许可证要求。
