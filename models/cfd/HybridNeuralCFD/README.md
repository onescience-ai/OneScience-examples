<p align="center">
  <strong>
    <span style="font-size: 30px;">HybridNeuralCFD</span>
  </strong>
</p>

# 模型介绍

本模型复现论文《Differentiable Hybrid Neural-CFD Modelling of Wall-Bounded Turbulence: Coupled Learning of Subgrid-Scale and Wall Closures》（arXiv:2607.17357）。

在可微分不可压有限体积求解器内，端到端联合学习亚格子尺度（SGS）闭合与壁面闭合，仅使用湍流统计量训练，用于粗分辨率壁湍流（零压力梯度湍流边界层，ZPG TBL）的壁面模化大涡模拟（WMLES）。

论文：Differentiable Hybrid Neural-CFD Modelling of Wall-Bounded Turbulence: Coupled Learning of Subgrid-Scale and Wall Closures
https://arxiv.org/abs/2607.17357

# 模型描述

模型由三部分组成：

- **神经 SGS 闭合**（`model/closures_unet.py`、`model/closures_sgs.py`）：3D U-Net 卷积结构预测亚格子应力，并融合经典 Vreman 模型作为物理先验。
- **神经壁面闭合**（`model/closures_wall.py`）：2D U-Net 预测壁面应力，结合解析 scaling law。
- **可微分有限体积求解器**（`model/solver_finite_volume.py`）：不可压分步投影求解器，贯穿前向传播与反向传播，实现端到端可微。

框架为 JAX/Flax，权重格式为 msgpack（flax.serialization）。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用合成湍流统计量 GT 数据端到端训练 SGS 与壁面闭合 |
| 模型推理 | 加载训练权重进行冻结闭合长期 rollout 推理与壁湍流统计量预测 |
| 本地快速验证 | 使用小网格运行 6 项全路径冒烟测试（forward/backward/train/val/metric/config） |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |

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
modelscope download --model OneScience/HybridNeuralCFD --local_dir ./model
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

当前复现使用合成湍流统计量 GT（synthetic-GT，ASSUMPTION），用于 smoke 调试与端到端训练验证，对应零压力梯度湍流边界层（ZPG TBL）。完整 DNS/WRLES 参考数据（Ground Truth）在当前复现中不可得。

### 训练

```bash
python scripts/main_train.py --config conf/config.yaml --output-dir ./out --max-epochs 10
```

训练会在指定输出目录（如 `out/checkpoints/`）下保存 msgpack 权重（`final.msgpack`）。

### 训练权重

已上传权重：`weight/final.msgpack`（tier_1 快速复现 20 epochs，loss 13.37 -> 11.08）。
推理指标（pipeline 验证值，非论文定量）：C_f=1.0396, mean_velocity=0.2889, rms_u1=4.5710。

### 推理

```bash
python scripts/main_infer.py --config conf/config.yaml --checkpoint ./weight/final.msgpack --output-dir ./out_eval
```

推理结果（平均速度剖面、r.m.s. 脉动、摩擦系数 C_f 等）会保存至输出目录。

### 评估和可视化

```bash
python scripts/test_smoke.py --config conf/config.yaml
```

运行 6 项全路径冒烟测试（forward/backward/train_loop/validation_loop/domain_metric C_f/config_consistency）验证模型正确性。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为以下论文的复现版本：

- Differentiable Hybrid Neural-CFD Modelling of Wall-Bounded Turbulence: Coupled Learning of Subgrid-Scale and Wall Closures（arXiv:2607.17357，J. Fluid Mech. 审稿中）
  https://arxiv.org/abs/2607.17357

> 复现说明：由于 Diff-FlowFSI 可微分求解器内部、DNS/WRLES GT 数据、U-Net 具体层结构及初始场生成方法在论文/公开资料中不可得，本复现为基于论文方法描述的独立实现，训练与推理指标为 pipeline 验证值，不等同于论文定量结果。
