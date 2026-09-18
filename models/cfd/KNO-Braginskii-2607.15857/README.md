<p align="center">
  <strong>
    <span style="font-size: 30px;">KNO-Braginskii-2607.15857</span>
  </strong>
</p>

# 模型介绍

KNO-Braginskii-2607.15857 是基于电阻率条件化 Koopman 神经算子（Resistivity-conditioned Koopman Neural Operator, KNO）的漂移约化 Braginskii 湍流场代理模型，复现自论文《Surrogate modeling of drift-reduced Braginskii turbulence with resistivity-conditioned Koopman neural operators》（arXiv:2607.15857）。

模型对三维边界等离子体湍流的关键诊断场进行短时程场演化预测，电阻率标签 ν0 直接嵌入潜在 Koopman 演化算子，实现跨电阻率区间的插值代理建模。包含 4 个独立 fieldwise 模型：密度 θ=ln n、电子温度 Te、电势 ϕ（strmf）、涡度 Ω（omega）。

论文：https://arxiv.org/abs/2607.15857

# 模型描述

KNO 采用"时间编码 → 空间编码 → 谱 Koopman 演化 → 空间解码 → 时间解码"架构：

- `temporal_encoder.py`：Etemp，逐网格点线性层 (2→32) + tanh
- `spatial_encoder.py`：E3D/D3D，3×Conv3d (32,128,32)
- `spectral_koopman.py`：方向谱算子 Kd(ν0,Δt)=K0+Kν·ν0+Kdt·Δt，FFT→R→Z→φ→IFFT，残差更新 Niter=2
- `temporal_decoder.py`：Dtemp，线性层 (32→4)，输出 4 帧预测
- `knn_fieldwise.py`：KNOFieldwiseModel 全流程组装

训练损失 L = 5·Lpred + 0.5·Laux（Lpred=MSE 4 帧预测，Laux 含辅助预测头与潜在一致性 4 项）。优化器 Adam，lr=1e-3，StepLR 每 500 epoch ×0.8，weight decay 1e-4。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用合成 Braginskii-like 湍流场数据训练 fieldwise KNO |
| 模型推理 | 加载权重对目标场做单步/自回归 rollout 预测 |
| 评估 | 计算 MSE/R²/相对 L2/谱斜率/压力梯度尺度长等诊断指标 |
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
modelscope download --model OneScience/KNO-Braginskii-2607.15857 --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy pyyaml matplotlib
```

**GPU环境**
```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy pyyaml matplotlib
```

### 训练数据介绍

（请在此处说明训练数据来源和获取方式）论文数据来自 GBS（Global Braginskii Solver）模拟；本复现使用可复现 seed 的合成 Braginskii-like 湍流场（`model/synthetic_data.py`）。

### 训练

```bash
python scripts/train.py --config conf/config.yaml --field theta --max-epochs 400 --device cuda
```

训练会在 `checkpoints/` 下保存 `best_{field}.pt`。

### 训练权重

- `weight/best_theta.pt`：密度 θ=ln n 模型
- `weight/best_temperature.pt`：电子温度 Te 模型
- `weight/best_strmf.pt`：电势 ϕ 模型
- `weight/best_omega.pt`：涡度 Ω 模型

### 推理

```bash
python scripts/infer.py --config conf/config.yaml --field theta --checkpoint weight/best_theta.pt --rollout 12
```

推理结果会保存至 `outputs/`。

### 评估和可视化

```bash
python scripts/eval.py --config conf/config.yaml --field theta --checkpoint weight/best_theta.pt --rollout-steps 12
```

评估结果保存至 `outputs/eval_result_{field}.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 arXiv:2607.15857 原始论文的复现版本：Surrogate modeling of drift-reduced Braginskii turbulence with resistivity-conditioned Koopman neural operators，https://arxiv.org/abs/2607.15857。
- 许可协议：Apache License 2.0。

