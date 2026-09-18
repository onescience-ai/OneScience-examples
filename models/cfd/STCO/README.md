<p align="center">
  <strong>
    <span style="font-size: 30px;">STCO</span>
  </strong>
</p>

# 模型介绍

STCO（Spatiotemporal Conditional Operator，时空条件神经算子）是面向"规定条件算子学习"（Prescribed-Condition Operator Learning, PCOL）的通用条件接口，用于时间依赖偏微分方程（PDE）的未来状态预测。它将观测历史、前导时间与目标时刻规定条件场（几何运动、入流扰动、体积力）映射到未来响应场。其条件接口结合 Flow-Aware Graph Leaf（FAGL）与 Dual-Site Feature-wise Linear Modulation（DSFiLM），可适配异构骨干架构。

论文：STCO: Conditional Neural Operators for Time-Dependent PDEs (arXiv:2608.20477)
https://arxiv.org/abs/2608.20477

本仓库为论文复现版本，聚焦 STCO 条件接口（FAGL + DSFiLM）的实现与验证。由于论文基准数据（WaterLily.jl 142 个 moving-body CFD 模拟）未公开，本复现采用简化的 moving-body 流动数据演示条件接口的增益逻辑。

# 模型描述

STCO 实现 FAGL + DSFiLM 条件接口，并以图消息传递（Graph Message-Passing）算子作为代表骨干：

- `stco.py`：STCO 管线（输入适配器 E_b + IN-DSFiLM + 骨干核心 + OUT-DSFiLM + 稠密读出 D_b）
- `fagl.py`：Flow-Aware Graph Leaf，涡量感知自适应槽位划分 + IDW4 空间对齐
- `dsfilm.py`：Dual-Site Feature-wise Linear Modulation（运动/入流/力三路由 + 前导缩放 + 门控）
- `mgn.py` / `fno.py`：图消息传递骨干 / FNO 骨干

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 规定条件算子学习 | 从观测历史与目标时刻规定条件场预测未来速度-压力场 |
| 时间依赖 PDE 未来预测 | 在二维不可压缩 moving-body 流动上验证条件接口增益 |
| 模型训练 | 使用 `python scripts/main.py train --mode stco` 训练 STCO 配置 |
| 模型评估 | 使用 `python scripts/main.py eval --mode stco` 输出 E_y / MCF 指标 |
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
modelscope download --model OneScience/STCO --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install numpy==1.26.4 torch==2.5.1 scipy==1.14.1 PyYAML==6.0.3 -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install numpy==1.26.4 torch scipy==1.14.1 PyYAML==6.0.3
```

### 训练数据介绍

（请在此处说明训练数据来源和获取方式）

论文基准数据（WaterLily.jl moving-body CFD）未公开。本仓库提供简化数据生成器，使用 `python scripts/main.py gen-data` 生成复现数据（60 个模拟，64×64 网格，256 帧，7 类条件族）。

### 训练

```bash
cd scripts
python main.py gen-data --cfg ../conf/stco_pino.yaml
python main.py train --mode stco --cfg ../conf/stco_pino.yaml
python main.py train --mode base --cfg ../conf/stco_pino.yaml
```

训练会在 `checkpoints/{mode}/` 下保存 `stco_{mode}_best.pt`。

### 训练权重

- `weight/STCO_stco_best.pt`：STCO 配置最佳权重（含 FAGL + DSFiLM 条件接口）
- `weight/STCO_base_best.pt`：Base 配置最佳权重（跳过 DSFiLM，条件接口关闭）

### 推理

```bash
cd scripts
python main.py eval --mode stco --ckpt ../weight/STCO_stco_best.pt --cfg ../conf/stco_pino.yaml
```

推理结果（E_y 场误差、MCF 条件敏感性）会保存至 `results/{mode}/`。

### 评估和可视化

```bash
cd scripts
python main.py eval --mode base --ckpt ../weight/STCO_base_best.pt --cfg ../conf/stco_pino.yaml
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 STCO 论文（arXiv:2608.20477）的复现版本。

- 本仓库为 STCO: Conditional Neural Operators for Time-Dependent PDEs 原始论文的复现版本。
- Xingxin Yang, Zhan Zhang, Juan Li. STCO: Conditional Neural Operators for Time-Dependent PDEs. arXiv:2608.20477.