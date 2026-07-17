
<p align="center">
  <strong>
    <span style="font-size: 30px;">Transolver-Airfoil-Design</span>
  </strong>
</p>

# 模型介绍

Transolver-Airfoil-Design 是基于清华大学 THUML 团队提出的 Transolver 构建的二维翼型外流场预测模型，适配 AirfRANS 非结构网格翼型数据。其核心思想是将离散网格自适应划分为可学习、物理状态相似的切片，再通过切片间注意力建模复杂物理相关性，最终映射回原始网格节点以预测流场变量。在本项目中，模型以翼型外形、来流条件、几何距离和法向量等节点特征为输入，预测非结构网格上的速度、压力和湍流黏度等变量，可用于翼型气动性能快速评估、CFD 代理建模、设计空间搜索、气动外形优化和大规模仿真加速等场景。

# 仓库说明

本仓库是 OneScience 整理的 Transolver-Airfoil-Design 标准运行包，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：
* 生成轻量假数据用于流程连通性验证
* 训练
* 推理
* 评估与可视化

当前不支持能力：
* 不内置预训练权重
* 不负责自动下载、清洗或重新适配全部外部数据库

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 翼型气动设计 | 快速预测二维翼型外流场，用于候选外形筛选 |
| 设计空间搜索 | 支持大量翼型设计方案的快速评估 |
| 工业仿真加速 | 面向复杂几何 PDE 代理求解和大规模仿真加速 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | OneCode 元信息 | 保持最小配置 |
| `conf/config.yaml` | 训练、推理和数据配置 | 默认使用轻量 fake AirfRANS 数据 |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成 PyG 图样本和归一化统计量 |
| `scripts/train.py` | 训练脚本 | 支持 fake 数据和真实 AirfRANS 数据 |
| `scripts/inference.py` | 推理脚本 | 读取训练权重并保存预测结果 |
| `scripts/result.py` | 结果查看脚本 | 打印推理指标摘要 |
| `model/` | 模型文件包 | OneScience复现的经典TOP模型 |
| `weight/` | 权重目录 | 训练默认保存 `Transolver.pth` |

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

### 安装运行环境

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 生成假数据进行流程验证

默认配置使用 `datapipe.backend: fake_airfrans`，可直接生成最小图数据检查训练和推理链路。

```bash
python scripts/fake_data.py
```

OneScience 社区提供可供训练的 `airfrans` 数据，用户可通过下述命令下载，并确认 `config/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/airfrans --local_dir ./data
```
### 训练

```bash
python scripts/train.py
```

训练参数来自 `conf/config.yaml` 的 `model`、`datapipe` 和 `training` 字段。默认配置用于最小 smoke test；真实训练时请将 `datapipe.backend` 改为 `airfrans`，并将 `datapipe.source.data_dir` 指向真实 AirfRANS `data/` 目录。

默认训练会保存 checkpoint：

```text
./weight/Transolver.pth
```

### 推理

```bash
python scripts/inference.py
```

推理会读取 `training.checkpoint_dir` 下的模型权重，默认是：

```text
./weight/Transolver.pth
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

- Transolver 原始论文：[Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366)。
- 本仓库保留来源说明，并面向 OneScience 自动运行场景进行整理。
