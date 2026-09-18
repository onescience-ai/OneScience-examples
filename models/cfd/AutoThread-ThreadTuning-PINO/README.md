<p align="center">
  <strong>
    <span style="font-size: 30px;">AutoThread-ThreadTuning-PINO</span>
  </strong>
</p>

# 模型介绍

AutoThread-ThreadTuning-PINO 是论文《Hybrid-Adaptive Thread Tuning to Mitigate Simulation Execution Bottlenecks in High-Performance Reinforcement Learning Inference》(arXiv:2608.06025) 的复现模型包。该模型用于在仿真在环强化学习（SiL-RL）推理中预测 DES（离散事件仿真）模拟端的最优工作线程（WT）数，从而缓解模拟执行瓶颈、提升系统吞吐。

核心方法 AutoThread 由两部分组成：
1. **PINO 预测器**（Physics-Informed Neural Operator）：以 7 维系统状态特征 `a={N, mu_e, k_obs, usr_cpu, system_cpu, VSZ, cswch}` 为输入，预测最优 WT 数 `k*` 与竞争系数 `(a1, b1)`，并用有限源 M/M/1 排队模型（公式 Eq.1）作为物理约束指导训练。
2. **自适应动态调优器**（Algorithm 1）：负载感知快速预测 + CPU 感知周期调整 + 吞吐驱动微调的三级递进闭环调整。

# 模型描述

模型结构为 PINO 多输出头交叉注意力架构：
- 应用特征编码器（MLP，输入 `(N, mu_e, k_obs)`）
- 硬件特征编码器（MLP，输入 `(usr_cpu, system_cpu, VSZ, cswch)`）
- 交叉注意力融合模块（1 头）
- FC 解码头输出 `k*`（SoftPlus，取整为线程数）与竞争系数 `(a1, b1)`
- 物理损失：`L_phy = ||T(N,k,mu_b,mu_e) - T_obs||^2 + alpha*||mu_b/mu_e - k*||^2`
- 总损失：`L = L_phy + lambda*L_sup + gamma*||theta||^2`（含 RBA 残差驱动权重）

主要模型文件：
- `autothread/models/pino_predictor.py` — PINO 网络定义
- `autothread/losses/autothread_loss.py` — 物理约束损失（含 P0 与分段模型）
- `autothread/data/synthetic_workload.py` — 合成工作量数据生成（Eq.1 + 竞争模型）
- `autothread/tuner/dynamic_tuner.py` — Algorithm 1 动态调优器

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 线程数预测 | 给定系统状态特征，预测最优工作线程数 k* |
| RL 推理模拟加速 | 在动态负载下闭环调整 WT 数以提升吞吐 |
| 模型训练 | 使用合成工作量数据训练 PINO 预测器 |
| 模型评估 | 测试集 k* 预测误差（RMSE/MAE）与 Algorithm 1 吞吐对比 |

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
modelscope download --model OneScience/AutoThread-ThreadTuning-PINO --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[general-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[general-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

训练使用合成工作量数据（由 `scripts/generate_data.py` 生成，500 样本，0.7/0.2/0.1 划分）。真实 DES 多线程轨迹数据集来自论文作者开源仓库，本复现包不附带。

```bash
python scripts/generate_data.py --config conf/default.yaml
```

### 训练

单卡（DCU/GPU）：

```bash
python scripts/train.py --config conf/default.yaml
```

训练会在 `weight/` 下保存 `best_model.pt` 与 `final_model.pt`。

### 训练权重

已上传权重：
- `weight/best_model.pt` — 验证集最优模型
- `weight/final_model.pt` — 最终模型

### 推理

```bash
python scripts/evaluate.py --config conf/default.yaml --ckpt weight/best_model.pt
```

### 评估和可视化

```bash
python scripts/run_autothread.py --config conf/default.yaml --ckpt weight/best_model.pt
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 arXiv:2608.06025 论文的复现版本，论文信息：

- 标题：Hybrid-Adaptive Thread Tuning to Mitigate Simulation Execution Bottlenecks in High-Performance Reinforcement Learning Inference
- 作者：Jiming Su, Hantao Hua, Lujia Yin, Yiping Yao, Feng Zhu
- arXiv: 2608.06025 (https://arxiv.org/abs/2608.06025)
- License: Apache License 2.0

