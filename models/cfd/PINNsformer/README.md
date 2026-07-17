<p align="center">
  <strong>
    <span style="font-size: 30px;">PINNsformer</span>
  </strong>
</p>

# 模型介绍

PINNsformer 是一种基于 Transformer 的物理信息神经网络框架，用于近似求解偏微分方程（PDE）。它将传统 PINN 的点式输入转换为伪时间序列，并通过时空嵌入、编码器-解码器注意力结构、Wavelet 激活函数和序列化物理约束损失，增强模型对时变 PDE 中时间依赖关系的刻画能力。

# 仓库说明

本仓库是 OneScience 整理的 PINNsformer 最小可运行标准工程，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

* 生成一维反应方程假数据
* 训练 PINNsformer1D
* 加载训练权重进行推理
* 评估相对 L1/L2 误差并生成可视化图

当前不支持能力：

* 不内置预训练权重
* 不内置对流方程、Navier-Stokes 等 `.mat` 数据示例
* 不提供多卡分布式训练脚本

# 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 时变 PDE 求解 | 通过物理残差、边界条件和初始条件约束训练连续场代理模型 |
| 物理信息神经网络验证 | 快速检查 PINNsformer 网络、损失函数、权重保存和推理链路 |
| 一维反应方程示例 | 基于解析解生成目标场，用于流程连通性验证和误差计算 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 训练、推理、数据和输出路径配置 | 已适配本仓库相对路径 |
| `model/pinnsformer.py` | PINNsformer 模型文件 | OneScience复现的经典TOP模型 |
| `scripts/common.py` | 脚本公共工具 | 配置读取、路径解析、模型构建、解析解与误差计算 |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成 `data/reaction_fake.npz` |
| `scripts/train.py` | 训练脚本 | 保存 `weight/1dreaction_pinnsformer.pt` 和 `result/train_loss.npy` |
| `scripts/inference.py` | 推理脚本 | 读取权重并保存 `result/prediction.npz` |
| `scripts/result.py` | 评估与可视化脚本 | 保存 `result/metrics.json` 和 `result/1dreaction_pinnsformer.png` |
| `weight/` | 权重目录 | 默认仅保留 `.gitkeep`，训练后生成权重 |
| `data/` | 数据目录 | 默认仅保留 `.gitkeep`，假数据脚本会生成数据 |
| `result/` | 结果目录 | 默认仅保留 `.gitkeep`，推理和评估后生成结果 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

* CPU 可用于小配置流程验证。
* 推荐使用 GPU 或 DCU 进行较大网格、较多 epoch 的训练。
* DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


## 3. 快速开始


### 安装运行环境

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 生成假数据进行流程验证
如需先验证脚本、模型、checkpoint 和结果文件是否能够完整跑通，可使用仓库内置脚本生成最小数据。

```bash
python scripts/fake_data.py
```
如需使用真实数据，可通过如下链接下载，并将 `conf/config.yaml` 中的 `data.data_dir` 进行正确的路径配置。

| 下载源 | 链接 | 提取码 | 下载后放置位置 |
|---|---|---|---|
| 百度网盘 | https://pan.baidu.com/s/1pM4ICc6FJX5pLF7WEoozxQ?pwd=5gha | `5gha` | `convection/convection.mat` 和 `navier_stokes/cylinder_nektar_wake.mat` |

### 训练

```bash
python scripts/train.py
```
默认配置使用较小网格和较少 LBFGS 迭代，便于快速跑通流程。如需恢复原始示例规模，可修改 `conf/config.yaml`：

```yaml
data:
  x_num: 101
  t_num: 101

training:
  epochs: 500
```

### 推理

```bash
python scripts/inference.py
```

推理会读取 `weight/1dreaction_pinnsformer.pt`，并保存：

```text
result/prediction.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估会保存：

```text
result/metrics.json
result/1dreaction_pinnsformer.png
```


# 配置说明

核心配置位于 `conf/config.yaml`：

| 配置段 | 说明 |
| :--- | :--- |
| `runtime` | 随机种子和默认设备选择 |
| `model` | PINNsformer1D 网络宽度、隐藏维度、层数和注意力头数 |
| `data` | 一维反应方程的空间/时间范围、网格数量和伪序列长度 |
| `equation` | 反应项系数和初始条件参数 |
| `training` | 训练 epoch、优化器和权重路径 |
| `paths` | 假数据、预测、指标、loss 和图像输出路径 |


# 输出说明

| 输出路径 | 说明 |
| :--- | :--- |
| `weight/1dreaction_pinnsformer.pt` | 训练后的模型权重 |
| `result/train_loss.npy` | 每个 epoch 的 residual、boundary、initial 和 total loss |
| `result/prediction.npz` | 推理预测、解析解、绝对误差和相对误差 |
| `result/metrics.json` | 相对 L1/L2 误差 |
| `result/1dreaction_pinnsformer.png` | 预测场、解析解和绝对误差可视化 |


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

* PINNsformer 原始论文：[PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks](https://arxiv.org/abs/2307.11833)。
* 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
