<p align="center">
  <strong>
    <span style="font-size: 30px;">SINO</span>
  </strong>
</p>

# 模型介绍

SINO（Starter-Iterator Neural Operator）是一个统一的谱-时空协同算子学习框架，用于高保真 PDE 前向仿真与逆向重建。SINO 通过神经网络重释经典定点迭代方法 Au=f 的初始化策略与迭代格式：频域初始化模块（Starter，FNO 型谱块）捕获全局稳定的低频特征，时域迭代学习模块（Iterator，CNN 卷积残差细化）优化局部解残差，配合多尺度 V-cycle 与时间变量正弦嵌入，有效克服单一域建模的精度瓶颈与长时间序列不稳定问题。

论文：Starter-Iterator Neural Operator: A Unified Architecture for High-Fidelity Forward and Inverse PDE Problems
https://arxiv.org/abs/2606.18305

本仓库复现论文在 1D Burgers 方程上的零样本分辨率泛化实验：训练于 R=256，可直接在 R=512..8192 分辨率上评估，误差保持稳定并优于 FNO baseline。

# 模型描述

SINO 采用 lifting-operator-projection 范式，核心组件如下（见 model/ 目录）：

- `sino.py` — SINO 主模型（Starter + Iterator + 多尺度 V-cycle + 时间嵌入 + Lifting/Projection）
- `starter.py` — Starter 谱块（FFT → 低频截断 → 可学习谱权重 → IFFT，分辨率自适应）
- `iterator.py` — Iterator CNN 迭代残差细化（unrolled N 步固定点迭代）
- `multiscale.py` — 多尺度 V-cycle（restriction strided conv / prolongation transposed conv）
- `time_embedding.py` — 正弦时间嵌入（log-spaced 频率 + MLP 投影）
- `fno.py` — FNO baseline（对比实验用）
- `loss.py` — 相对 L2 损失（论文 Eq.13）

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| PDE 算子学习训练 | 使用 Burgers 谱解数据训练 SINO 或 FNO baseline |
| 零样本分辨率泛化评估 | 训练于 R=256，直接评估 R=256..8192 的相对 L2 误差 |
| 单步相对 L2 评估 | 在测试集上计算单步预测相对 L2 误差 |
| 自回归 rollout | 长时间序列递归预测（时间演化任务） |


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

# 根据当前仓库自行设置
```bash
modelscope download --model OneScience/SINO --local_dir ./model
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

训练数据为 1D Burgers 方程谱解，本地生成（`scripts/generate_burgers.py`）。样本为初始条件 u0 → 解 u(x,T)，600 样本，R=256，nu=0.001。

```bash
python scripts/generate_burgers.py --out ./data/burgers_r256_train.pt --num-samples 600 --resolution 256 --dt 0.0005
```

### 训练

```bash
python scripts/train.py --config conf/burgers.yaml --model sino --epochs 50 --checkpoint-dir ./checkpoints
```

### 训练权重

- `weight/sino_best_model.pt` — SINO 模型权重（Burgers 零样本分辨率泛化，test_rel_l2≈0.061）
- `weight/fno_best_model.pt` — FNO baseline 权重（test_rel_l2≈0.084）

### 推理

```bash
python scripts/roll_out.py --config conf/burgers.yaml --checkpoint ./weight/sino_best_model.pt --steps 1 --out ./outputs/rollout_preds.npy
```

### 评估和可视化

```bash
python scripts/evaluate.py --config conf/burgers.yaml --checkpoint ./weight/sino_best_model.pt --out ./outputs/eval_sino.json
python scripts/zero_shot_resolution.py --config conf/burgers.yaml --checkpoint ./weight/sino_best_model.pt --resolutions 256,512,1024,2048,4096,8192 --out ./outputs/zero_shot_sino.json
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为 Starter-Iterator Neural Operator (SINO) 原始论文的复现版本。
- 论文：Starter-Iterator Neural Operator: A Unified Architecture for High-Fidelity Forward and Inverse PDE Problems, arXiv:2606.18305, 2026.

