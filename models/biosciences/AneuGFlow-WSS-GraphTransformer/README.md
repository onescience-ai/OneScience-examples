<p align="center">
  <strong>
    <span style="font-size: 30px;">AneuGFlow-WSS-GraphTransformer</span>
  </strong>
</p>

# 模型介绍

本模型是论文《Real-Time Pulsatile Flow Prediction for Realistic, Diverse Intracranial
Aneurysm Morphologies using a Graph Transformer and Steady-Flow Data Augmentation》
（arXiv:2601.19876，领域：CFD / 生物医学血流力学）的复现实现。

它使用 **GPS-style Graph Transformer**（GINE 局部消息传递 + 全局多头注意力）加上
**GHD 几何编码** 与 **1D 卷积 U-Net 波形编码**，从颅内动脉瘤（IA）表面网格和入口
质量流量波形预测完整心动周期内的瞬态 **壁面剪切应力（WSS）矢量场**，并以
**稳态数据增强** 提升可泛化性，实现无需 CFD 的实时预测。

论文：Real-Time Pulsatile Flow Prediction for Realistic, Diverse Intracranial Aneurysm
Morphologies using a Graph Transformer and Steady-Flow Data Augmentation
https://arxiv.org/abs/2601.19876

> 数据说明：官方 AneuG-Flow 数据集仅托管于 HuggingFace (`whding123/AneuG-Flow`)，
> 在发布/复现环境不可达。本包附带 **合成等效数据管线**（模板网格 + GHD 变形 +
> 物理一致的 WSS 生成）用于端到端验证。**合成数据取得的指标不作声称对齐论文
> （论文 MSE 0.179 / SSIM 0.982 / rL2\* 2.84%）。** 提供加载真实 AneuG-Flow `.pth`
> 数据的接口（`data/dataset.py` 的 `real` 数据源），接入真实数据后即可对齐论文。

# 模型描述

模型结构（GPS-style Graph Transformer）：

- **几何编码（Table II，每节点 80 维）**：坐标(3) + 法向(3) + 节点类型 one-hot(2) +
  GHD shape modes(8) + 梯度(8) + 特征值(8) + cotangent-Laplacian 特征向量(16) +
  梯度(16) + 特征值(16)。
- **波形编码**：对入口质量流量脉冲（叠加其导数）做 1D 卷积 U-Net 编码，得到逐帧
  结构编码并注入每个 GPS block；稳态样本将其掩蔽为零（稳态数据增强）。
- **GPS block**：GINE 局部消息传递 + 全局多头自注意力 + 残差 + LayerNorm + FFN。
- **输出头**：MLP -> 每节点 WSS 矢量 `[N,3]`；损失 = MSE。

主要模型文件：`model/model.py`、`model/gps.py`、`model/waveform.py`、
`model/geometry.py`、`model/graph_ops.py`、`model/dataset.py`。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 血流动力学训练 | 使用 IA 表面网格 + 质量流量波形训练瞬态 WSS 预测 |
| 本地快速验证 | 使用合成数据检查数据读取、模型训练与推理、指标计算 |
| 端到端管线验证 | 合成 fallback 数据驱动完整训练 -> checkpoint -> 评估闭环 |
| 真实数据迁移 | 接入官方 AneuG-Flow `.pth` 数据后可直接对齐论文指标 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本。

### 下载模型包

```bash
modelscope download --model OneScience/AneuGFlow-WSS-GraphTransformer --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem、bio、all（全领域）
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本包默认使用合成 fallback 数据管线（模板 UV 球体网格 + GHD 变形 + 物理一致 WSS），
在运行时由 `scripts/make_synthetic_data.py` 生成。接入真实 AneuG-Flow 数据时，将官方
`.pth` 文件放入 `data/real_datasets/` 并在配置中设置 `data.data_source: real`。

### 训练

```bash
# Tier0 冒烟测试（验证代码正确性）
python scripts/smoke_test.py

# 生成合成数据集快照
python scripts/make_synthetic_data.py --out data/synthetic

# 训练（天数据集端到端验证）
python scripts/train.py --config conf/train_medium.yaml
```

训练会在 `checkpoints/` 下保存验证损失最小的 `best.pt`。

### 训练权重

即将上传（`weight/best.pt`，在合成数据上训练，验证损失 ~0.0016）。

### 推理

本模型以 `evaluate.py` 提供推理与指标评估（模型 forward 在
`model/model.py` 的 `WSSGraphTransformer.forward`）。

```bash
# 载入 checkpoint 并在测试集上评估（含 MSE/rL2/rL2*/SSIM/SSIMr）
python scripts/evaluate.py --config conf/train_medium.yaml --checkpoint weight/best.pt --split test
```

### 评估和可视化

```bash
python scripts/evaluate.py --config conf/train_medium.yaml --checkpoint weight/best.pt --split test
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为论文《Real-Time Pulsatile Flow Prediction for Realistic, Diverse Intracranial
Aneurysm Morphologies using a Graph Transformer and Steady-Flow Data Augmentation》
（arXiv:2601.19876，CC BY-NC-SA 4.0）的复现版本。模型结构遵循 GPS Graph Transformer
方法；复现实现的代码遵循本仓库 Apache License 2.0。论文指标仅能在真实 AneuG-Flow
数据上复现，本包附带的合成数据不声称对齐论文数值。