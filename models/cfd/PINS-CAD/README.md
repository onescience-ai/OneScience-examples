<p align="center">
  <strong>
    <span style="font-size: 30px;">PINS-CAD</span>
  </strong>
</p>

# 模型介绍

PINS-CAD（Physics-Informed, Self-supervised learning framework for predictive modeling of Coronary Artery Digital twins）是一个物理信息自监督学习框架。它在合成冠状动脉数字孪生（200,000 个血管树）上预训练图神经网络，预测压力与流量分布，预训练受 1D Navier–Stokes 方程与压降定律引导（无需 CFD 模拟与标签数据）；微调于临床数据后可用于预测未来心血管事件。

论文：Physics-informed self-supervised learning for predictive modeling of coronary artery digital twins
arXiv: https://arxiv.org/abs/2512.03055

本仓库为论文的 Tier 1 复现版本（合成数据小规模端到端）。

# 模型描述

- 层级 GNN 编码器：3 个 block × 4 层 GCN，Top-K pooling（ratio=0.5），特征维度 d→2d→3d。
- Centerline Aggregation (CA)：对每个 centerline 点聚合 Kca 个最近图节点特征（含相对坐标），经卷积与均值池化。
- 预测头：4 层线性 MLP 将 3d 维中心线特征降到 2 维，输出流量 Q 与压力 P。
- 物理损失：1D NS 残差损失 + 全局/滑动窗口局部压降一致性损失（Eq.4-9）。
- 下游微调：encoder 冻结，中心线特征全局池化后经分类 MLP 预测心血管事件（AUC）。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型预训练 | 使用合成冠状动脉血管树数据，以 1D NS + 压降定律物理损失自监督训练 |
| 模型微调 | 冻结 encoder，微调分类头预测心血管事件 |
| 模型推理 | 加载权重预测压力/流量/FFR 曲线 |
| 模型评估 | 物理损失 + AUC/Accuracy/F1 指标评估 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/PINS-CAD --local_dir ./model
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
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem（全领域）
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本模型使用 A3M（Anatomy-aware Augmentation Module）风格合成的冠状动脉数字孪生：程序化 3D 中心线基模 + 半径分布（含 stenosis）+ 几何增强（旋转/弯曲/高斯平滑）+ 截面扫描构建血管树图。数据生成器位于 `model/data/generate_synthetic.py`。

### 训练

自监督预训练（physics-informed）：

```bash
PYTHONPATH=model python scripts/train_pretrain.py --config conf/config.yaml
```

预训练权重保存至 `checkpoints/pretrain.pth`。

下游微调：

```bash
PYTHONPATH=model python scripts/train_finetune.py --config conf/config.yaml
```

微调权重保存至 `checkpoints/finetune.pth`。

### 训练权重

- `weight/pretrain.pth`：自监督预训练权重（200 棵合成血管树 × 200 epochs）。
- `weight/finetune.pth`：微调分类头权重（best_val_auc=0.8452，合成数据）。

### 推理

```bash
PYTHONPATH=model python scripts/infer.py --config conf/config.yaml
```

推理输出压力 P、流量 Q 与 FFR 曲线（`outputs/inference_results.json`, `outputs/ffr_curve.npy`）。

### 评估和可视化

```bash
PYTHONPATH=model python scripts/eval.py --config conf/config.yaml
```

评估输出物理损失（1D NS residual / 全局压降 / 局部压降）与下游指标（AUC / Accuracy / F1），保存至 `outputs/metrics.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为《Physics-informed self-supervised learning for predictive modeling of coronary artery digital twins》(arXiv:2512.03055) 的 Tier 1 复现版本。
- 模型输出为合成数据训练的演示结果，未使用真实 FAME2 临床数据。

