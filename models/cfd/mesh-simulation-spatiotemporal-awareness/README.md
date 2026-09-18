<p align="center">
  <strong>
    <span style="font-size: 30px;">mesh-simulation-spatiotemporal-awareness</span>
  </strong>
</p>

# 模型介绍

本模型复现 ICML 2026 论文《Mesh Based Simulations with Spatial and Temporal awareness》
(arXiv:2605.01542, cs.LG)。针对 CFD mesh-based surrogate 建模提出三项统一改进：

1. **Multi Node Prediction (MNP)**：stencil 级监督，让中心节点预测其 1-hop 邻居场值，强制空间导数一致性。
2. **Temporal Correction**：用 predictor-corrector + temporal Cross-Attention 替代显式 residual 更新，提升 long-horizon rollout 稳定性。
3. **3D RoPE**：对 Q/K 施加按坐标轴旋转的位置嵌入，捕捉非结构网格旋转对称性。

论文：Mesh Based Simulations with Spatial and Temporal awareness
https://arxiv.org/abs/2605.01542

# 模型描述

基于 Encode-Process-Decode 架构的 MeshGraphNet surrogate，在 NS 涡量时间序列网格图数据上训练（next-step 场预测 + 自回归 rollout）。支持三种架构（MeshGraphNet / Transformer / Transolver）与三项创新模块的正交组合。

主要模型文件：`model/code_src_models__meshgraphnet.py`、`model/code_src_models__mnp.py`、`model/code_src_models__temporal_correction.py`、`model/code_src_models__rope3d.py`。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用 NS 涡量网格图数据训练 surrogate（MNP + Temporal Correction + 3D RoPE） |
| 模型推理 | 加载权重进行 next-step 场预测与自回归 rollout |
| 模型评估 | 计算 1-step RMSE 与 All-Rollout RMSE |
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
modelscope download --model OneScience/mesh-simulation-spatiotemporal-awareness --local_dir ./model
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

训练使用 NS 涡量时间序列数据（替代论文官方 COMSOL Cylinder / CimLib Aneurysm，因本地不可得）。
数据格式：`u` (N, 64, 64, T) 涡量时间序列，构造 64×64 网格图（8 邻接，4096 节点），任务为 next-step 涡量场预测。

（请在此处补充公开数据集的下载地址）

### 训练

单卡：

```bash
python scripts/code__train.py --config conf/code_configs__tier1.yaml
```

训练会在 `repro_artifacts/2605.01542/checkpoints/seed0/` 下保存 `best.pt` 与 `last.pt`。

### 训练权重

- `weight/checkpoints_seed0__best.pt` — 最优验证 1-step RMSE checkpoint（约 2.3MB）
- `weight/checkpoints_seed0__last.pt` — 最后一个 epoch checkpoint（约 2.3MB）

### 推理

```bash
python scripts/code__rollout.py --config conf/code_configs__tier1.yaml --checkpoint weight/checkpoints_seed0__best.pt
```

推理结果（逐轨迹 rollout RMSE）会保存至 `logs/rollout_metrics.json`。

### 评估和可视化

```bash
python scripts/code__evaluate.py --config conf/code_configs__tier1.yaml --checkpoint weight/checkpoints_seed0__best.pt
```

评估产出 1-step RMSE 与 All-Rollout RMSE，保存至 `logs/metrics.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为《Mesh Based Simulations with Spatial and Temporal awareness》(arXiv:2605.01542, ICML 2026) 的复现版本。

