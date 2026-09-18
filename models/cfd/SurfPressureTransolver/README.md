<p align="center">
  <strong>
    <span style="font-size: 30px;">SurfPressureTransolver</span>
  </strong>
</p>

# 模型介绍

本模型复现论文《Evaluation of State-of-the-Art Deep Learning Architectures for Aerodynamical Predictions》(arXiv:2607.13866, https://arxiv.org/abs/2607.13866) 的 2D 翼型表面压力预测任务（Tier 1 小数据端到端）。

基于 RAE2822 翼型表面点，输入表面坐标、表面法向量、马赫数（Vinf 代理）与攻角，预测每点压力系数 C_p。主模型为 **Transolver**（Physics attention 变压器），并提供 **PointNet** 点云基线对比。

论文：Evaluation of State-of-the-Art Deep Learning Architectures for Aerodynamical Predictions
https://arxiv.org/abs/2607.13866

# 模型描述

- **Transolver**（主模型）：物理切片注意力 Transformer，将输入点按物理状态分组为 slices 后做自注意力，避免对海量网格点做二次方注意力。架构：linear lifting → 8 个 Transolver block（LayerNorm + Physics attention + FFN + 残差）→ linear decoder。
- **PointNet**（baseline）：逐点共享 MLP + 全局 max-pool 特征拼接，轻量点云基线。

Tier 1 复现结果（RAE2822 固定几何、变工况，600/200/200 划分，60 epochs）：

| 模型 | test Rel.L2 | MAE | R2 |
| :---: | :---: | :---: | :---: |
| Transolver | 0.0701 | 0.0276 | 0.9917 |
| PointNet | 0.3998 | 0.1675 | 0.7953 |

论文 2D 全量结果（Table 6）：Transolver Rel.L2 = 0.0225（本包为小数据近似复现）。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 表面压力预测训练 | 使用 RAE2822 翼型表面点数据训练 Transolver / PointNet |
| 模型推理 | 加载权重对表面点预测 C_p |
| 模型评估 | 计算 MAE/MSE/RMSE/Rel.L1/Rel.L2/R2 六项指标 |
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
modelscope download --model OneScience/SurfPressureTransolver --local_dir ./model
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

训练数据为 RAE2822 翼型表面 CFD 仿真数据（OneStore RAE2822_Dataset）。下载后置于 `data/` 目录，并在 `conf/default.yaml` 中配置 `data.path`。

### 训练

单卡：

```bash
python scripts/train.py --config conf/default.yaml
```

训练会在 `outputs/checkpoints/` 下按模型保存 `best_model.pt` 与 `last_model.pt`。

### 训练权重

`weight/` 下提供 Transolver 与 PointNet 的 best/last 权重（PyTorch .pt）：`transolver_best_model.pt`、`pointnet_best_model.pt` 等。

### 推理

```bash
python scripts/infer.py --config conf/default.yaml --split test
```

推理结果保存为 `outputs/results/pred_test.npy`。

### 评估和可视化

```bash
python scripts/eval.py --config conf/default.yaml
```

评估输出六项指标至 `outputs/results/metrics.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为 arXiv:2607.13866（Evaluation of State-of-the-Art Deep Learning Architectures for Aerodynamical Predictions, Scherz, Hines, Bekemeyer, DLR, 2026）论文复现版本，主要复现 Transolver 与 PointNet 在 2D 翼型表面压力预测上的 Tier 1 小数据结果。

