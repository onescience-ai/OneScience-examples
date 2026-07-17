<p align="center">
  <strong>
    <span style="font-size: 30px;">Functional Attention</span>
  </strong>
</p>

# 模型介绍

Functional Attention 是 Jiefang Xiao等人提出的神经算子注意力方法《Functional Attention: From Pairwise Affinities to Functional Correspondences》。该方法将传统 Transformer 中 token 之间的 pairwise attention 重新解释为函数空间中自适应基之间的 functional correspondence。模型通过学习查询/键值对应的自适应基函数，在低维基空间中求解带 Tikhonov 正则的线性算子，再映射回点空间，从而降低长序列和非结构网格上的注意力计算开销，并增强跨分辨率、跨离散方式的泛化能力。


## 模型架构

本工程采用“点级编码器 + 多层 Functional Attention Block + 点级回归头”的结构，直接在 AirfRANS 非结构网格点云上进行场预测。

| 阶段 | 输入/输出 | 作用 |
| --- | --- | --- |
| 输入特征 | `[x, y, u_inf_x, u_inf_y, sdf, normal_x, normal_y]` | 表示点坐标、来流条件、符号距离和表面法向 |
| 线性编码器 | `7 -> 256` | 将物理与几何特征映射到隐藏通道 |
| Functional Attention Blocks | `8` 层，`8` heads，`32` bases | 在自适应函数基空间中学习紧凑算子，并完成点间信息传递 |
| 归一化与输出头 | `256 -> 4` | 输出 `[u, v, p, nut]`，即二维速度、压力和湍流黏度 |

单个 Functional Attention Block 的主要计算步骤如下：

| 子模块 | 说明 |
| --- | --- |
| `LayerNorm` | 稳定隐藏特征分布 |
| `Q/K/V` 投影 | 生成查询、键和值特征 |
| 自适应基 `Phi/Psi` | 从当前点云特征中学习 query basis 和 key/value basis |
| 紧凑算子求解 | 在 `bases x bases` 空间中求解带 Tikhonov 正则的线性对应关系 |
| 点空间回映射 | 将基空间结果映射回原始点云 |
| 残差与 FFN | 保留原始特征并提升非线性表达能力 |

默认训练配置对齐论文 AirfRANS 设置：

| 配置项 | 默认值 |
| --- | --- |
| `layers` | `8` |
| `heads` | `8` |
| `channels` | `256` |
| `bases` | `32` |
| `loss` | `Lv + Ls` |
| `optimizer` | `Adam` |
| `epochs` | `400` |
| `learning_rate` | `1e-3` |

# 仓库说明
本仓库基于 OneScience 技能复现、面向论文中的 AirfRANS OOD Reynolds 实验场景，使用二维翼型 RANS 非结构网格数据，学习从翼型几何、来流条件、符号距离和表面法向到速度、压力、湍流黏度等物理场的映射。

当前支持能力：

- 训练
- 断点续训
- 推理
- 评估与指标导出
- 真实 AirfRANS 数据结构检查
- 训练集统计量读取与生成
- VTK XML 压缩数组解析
- e02r1n03 节点预检与训练辅助脚本

当前不支持能力：
- 不自动下载外部数据集
- 不提供官方 wall shear stress 和表面积分权重
- 不严格复现论文 Table 3 的升阻力积分口径，当前力系数指标为 pressure-only 近似
- 不包含官方预训练权重，`weight/funcattn_reynolds.pt` 是本地训练产物

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 二维翼型流场代理建模 | 在 AirfRANS 非结构网格上预测速度、压力和湍流黏度等点级物理场 |
| OOD Reynolds 泛化评估 | 使用 `reynolds_train -> reynolds_test` 验证跨雷诺数分布外泛化 |
| CFD 数值求解器代理加速 | 以神经算子近似 RANS 仿真结果，用于快速场预测和设计筛选 |
| 非结构网格注意力验证 | 验证 Functional Attention 的自适应基和紧凑算子在不规则网格上的建模能力 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | 元信息 | 定义数据检查、训练、推理和评估入口 |
| `config/config.yaml` | 数据、模型、训练和推理配置 | 默认 AirfRANS OOD Reynolds 配置 |
| `model/funcattn_airfrans/data/` | AirfRANS 数据读取与 VTK XML 解析 | 支持 manifest split、VTU/VTP 字段读取、缓存和归一化 |
| `model/funcattn_airfrans/models/` | Functional Attention 模型实现 | 包含 `FunctionalAttentionRegressor` |
| `scripts/data.py` | 数据检查脚本 | 读取 `manifest.json`，可生成训练集 mean/std |
| `scripts/train.py` | 训练脚本 | 支持读取 `weight/funcattn_reynolds.pt` 自动续训 |
| `scripts/inference.py` | 推理脚本 | 保存预测到 `.npz` |
| `scripts/result.py` | 评估脚本 | 输出 `weight/result_reynolds.json` |
| `scripts/preflight_e02r1n03.sh` | 节点预检脚本 | 检查 Python、Torch、设备和 checkpoint |
| `scripts/run_e02r1n03_train.sh` | 节点训练辅助脚本 | 在节点内启动训练 |
| `weight/` | 权重与结果目录 | 包含本地训练 checkpoint、训练日志和评估 JSON |

用户可通过魔搭社区下载预训练好的模型权重进行推理微调：
```
modelscope download --model OneScience/Functional_Attention  weight/funcattn_reynolds.pt --local_dir .
```

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU、DCU 或 HCU 运行完整训练。
- CPU 可用于导入检查和极小规模代码连通性验证。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


### 安装运行环境

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

## 3. 快速开始

### 数据集下载

用户可通过如下链接下载原始数据：

```text
https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip
```

### 数据检查

```bash
python scripts/data.py --config config/config.yaml
```

### 训练

```bash
python scripts/train.py --config config/config.yaml
```

默认训练配置：

- split: `reynolds_train -> reynolds_test`
- epochs: `400`
- batch size: `4`
- max points per case: `32000`
- checkpoint: `weight/funcattn_reynolds.pt`

训练脚本默认 `resume: true`，如果 checkpoint 已存在，会从已有权重继续训练。

### 推理和可视化

推理 3 个测试样本：

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/funcattn_reynolds.pt \
  --num-samples 3 \
  --output weight/predictions_reynolds.npz
```

当前推理脚本输出 `.npz` 预测文件，不生成图片。可视化可基于保存的点坐标和预测场自行绘制。

### 评估

```bash
python scripts/result.py \
  --config config/config.yaml \
  --checkpoint weight/funcattn_reynolds.pt
```

评估结果默认保存到：

```text
weight/result_reynolds.json
```

### 复现的实验设置

| 项目 | 当前工程设置 | 论文设置 |
| --- | --- | --- |
| 数据集 | AirfRANS 原始 `Dataset.zip` | AirfRANS |
| Split | `reynolds_train -> reynolds_test` | OOD Reynolds |
| 输入 | `[x, y, u_inf_x, u_inf_y, sdf, normal_x, normal_y]` | 几何、来流和边界信息 |
| 输出 | `[u, v, p, nut]` | 体/表面物理场 |
| Loss | `Lv + Ls` | `Lv + Ls` |
| Epoch | `400` | `400` |
| Optimizer | Adam | Adam |
| Learning rate | `1e-3` | `1e-3` |
| Model scale | 8 layers, 8 heads, 256 channels, 32 bases | 8 layers, 8 heads, 256 channels, 32 bases |

### 复现的结果对比

当前保留的本地训练结果位于 `weight/result_reynolds.json`。当前完成了可运行复现流程和 400 epoch 训练，但未达到论文主实验精度。主要差异来自力系数积分口径不完整、官方预处理和 `Lv + Ls` 细节未完全公开，以及本工程为独立重写实现。


# 引用与许可证

- Functional Attention 原始论文：[Functional Attention: From Pairwise Affinities to Functional Correspondences](https://arxiv.org/abs/2605.31559)
- AirfRANS 数据集论文：[AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions](https://arxiv.org/abs/2212.07564)
- AirfRANS 原始数据集下载：[Dataset.zip](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip)

- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理；公开分发前请根据上游项目和数据集确认许可证要求。
