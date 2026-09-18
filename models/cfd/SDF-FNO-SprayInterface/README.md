<p align="center">
  <strong>
    <span style="font-size: 30px;">SDF-FNO-SprayInterface</span>
  </strong>
</p>

# 模型介绍

SDF-FNO 是一个边界条件条件化的 Fourier Neural Operator 代理模型，用于预测喷雾气液界面的 signed distance function (SDF) 演化，来自论文《Towards Rapid Prototyping of Spray Injectors: A Regime-Agnostic Neural Operator Surrogate for Gas-Liquid Interface Evolution》(arXiv:2608.17825)。该模型在跨多个雾化流态的 2D sharp-interface Volume-of-Fluid CFD 仿真数据（本复现使用合成数据）上训练，支持自回归多步 rollout 预测，并可通过可微 SDF→相分数变换重建相分数场。

论文：https://arxiv.org/abs/2608.17825

# 模型描述

本模型基于 boundary-conditioned Fourier Neural Operator (FNO) 架构：
- 12 通道输入：4 帧历史归一化 SDF + 8 个静态/条件通道（坐标、归一化工况速度、入口/壁面/顶部 mask）
- 3 层 FNO 谱卷积 block（retained modes (16,8)，GELU 激活），latent width 32
- 输出头映射到 1 通道归一化 SDF
- 可微 SDF→相分数 proxy 重建（logistic + inventory 缩放）
- 训练损失：Lφ（SDF 重建）+ Lm（inventory 守恒）+ 0.05Lα（phase proxy）

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 喷雾界面演化预测 | 自回归预测气液界面 SDF 场演化 |
| 注入工况筛选 | 按 interfacial area per unit gas power 对工况排序 |
| 模型训练 | 使用合成/CFD 数据训练 SDF-FNO |
| 模型评估 | 计算 RMSEφ / RMSEαp / εm / IoU |

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
modelscope download --model OneScience/SDF-FNO-SprayInterface --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

论文原始 CFD 数据集不公开（仅向通讯作者索取）。本复现使用合成数据生成器（`scripts/synthesis.py`）生成等价验证数据。

```bash
python scripts/synthesis.py  # 或通过 train.py 自动生成
```

### 训练

```bash
python scripts/trainer.py
```

训练会保存 checkpoint 到 `outputs/checkpoints/`（best_model.pt / final_model.pt）。

### 训练权重

- `best_model.pt`（9.5 MB）
- `final_model.pt`（9.5 MB）

### 推理

```bash
<!-- 推理脚本未在 scripts/ 目录中找到，请补充 -->
```

### 评估和可视化

```bash
python scripts/evaluate.py
```

评估输出 RMSEφ / RMSEαp / εm / IoU（Δn=1,5,10）。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为论文《Towards Rapid Prototyping of Spray Injectors: A Regime-Agnostic Neural Operator Surrogate for Gas-Liquid Interface Evolution》(arXiv:2608.17825, physics.flu-dyn) 的复现版本。

- 论文: https://arxiv.org/abs/2608.17825

