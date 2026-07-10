<p align="center">
  <strong>
    <span style="font-size: 30px;">Transolver-Car-Design</span>
  </strong>
</p>

# 模型介绍

Transolver-Car-Design 是基于清华大学 THUML 团队提出的 Transolver / Transolver++ 构建的三维汽车外流场预测模型，适配 ShapeNetCar 非结构网格数据。模型将车辆几何相关的节点特征输入 Transolver 的 Physics-Attention 结构，通过物理切片建模全局流场关联，并在原始网格节点上预测三维速度和压力。

本工程中，输入特征为 7 维，包括三维坐标、SDF 和法向量；输出为 4 维，包括三维速度和压力。该模型可用于汽车外形方案快速评估、外流场代理建模、阻力系数相关分析和 CFD 仿真加速。

# 仓库说明

本仓库是 OneScience 整理的 Transolver-Car-Design 标准运行包，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：
* 生成轻量假数据用于流程连通性验证
* 训练 Transolver 或 Transolver_plus
* 推理并保存预测结果
* 查看推理结果摘要

当前不支持能力：
* 不内置完整 ShapeNetCar 数据集
* 不内置预训练权重
* 不负责自动下载、清洗或重新适配外部数据集

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 汽车气动设计 | 快速预测车辆外流场速度和表面压力 |
| CFD 代理建模 | 用神经网络近似复杂非结构网格上的流体求解过程 |
| 仿真加速 | 为大规模候选设计筛选提供轻量评估链路 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | OneCode 元信息 | 保持最小配置 |
| `conf/config.yaml` | 模型、数据、训练和推理配置 | 默认是 CPU 轻量 smoke test 配置 |
| `model/Transolver3D.py` | Transolver 3D 模型定义 | OneScience复现的经典TOP模型 |
| `model/Transolver3D_plus.py` | Transolver++ 3D 模型定义 | OneScience复现的经典TOP模型 |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成 ShapeNetCar 风格预处理 `.npy` 样本和统计量 |
| `scripts/train.py` | 训练脚本 | 读取 `conf/config.yaml` 并保存 checkpoint |
| `scripts/inference.py` | 推理脚本 | 读取 checkpoint，保存预测和真值 `.npy` |
| `scripts/result.py` | 结果查看脚本 | 打印预测文件数量和首个结果路径 |
| `weight/` | 权重目录 | 默认保存 `Transolver_plus.pth` |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**运行环境**

```bash
git clone https://gitee.com/onescience-ai/onescience.git
cd onescience
bash install.sh cfd
```

## 3. 快速开始

### 生成假数据进行流程验证

默认配置面向最小 smoke test。先生成 ShapeNetCar 风格的小型预处理数据：

```bash
python scripts/fake_data.py
```

### 训练

```bash
python scripts/train.py
```

训练参数来自 `conf/config.yaml` 的 `model`、`datapipe` 和 `training` 字段。默认配置使用轻量 `Transolver_plus` 参数，用于快速验证工程链路；真实训练时可将 `n_hidden`、`n_layers`、`n_head`、`slice_num` 等参数调回目标规模，并将数据路径指向真实 ShapeNetCar 数据。

默认训练会保存 checkpoint：

```text
./weight/Transolver_plus.pth
```

### 推理

```bash
python scripts/inference.py
```

推理会读取 `training.checkpoint_dir` 下的模型权重，默认是：

```text
./weight/Transolver_plus.pth
```

### 查看结果

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
- Transolver++ 相关工作请参考 THUML Transolver 项目和论文说明。
- 本仓库保留来源说明，并面向 OneScience 自动运行场景进行整理。
