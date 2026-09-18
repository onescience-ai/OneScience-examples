<p align="center">
  <strong>
    <span style="font-size: 30px;">RIBBO</span>
  </strong>
</p>

# 模型介绍

RIBBO（Reinforced In-Context Black-Box Optimization）是一个端到端从离线数据学习通用黑盒优化（BBO）算法的元学习方法。它采用 GPT 架构（因果 Transformer），输入优化历史三元组 (x, y, R)（x 为查询点、y 为函数值、R 为 regret-to-go token），输出下一个查询点的 diagonal Gaussian 分布。通过 regret-to-go tokens 与 Hindsight Regret Relabelling (HRR) 推理策略，RIBBO 能够自动识别任务与行为算法，并生成满足用户期望 regret 的查询点序列。

论文：Reinforced In-Context Black-Box Optimization
https://arxiv.org/abs/2402.17423

# 模型描述

RIBBO 基于 GPT（causal transformer）架构：
- Token 聚合：每个 (x, y, R) 三元组经 2 层 MLP 聚合为 256 维 token 嵌入（Concat 方法）
- Causal Transformer：12 层自注意力层、8 头、FFN 1024、dropout 0.1，保留位置编码
- 输出头：diagonal Gaussian 分布（均值 μ 与 log-std），预测下一个查询点
- 推理：Algorithm 1 自回归采样 + Hindsight Regret Relabelling（HRR）

核心超参（Table 1）：embedding 256、layers 12、heads 8、FFN 1024、dropout 0.1、batch 64、lr 2e-4、cosine annealing、子序列长度 τ=50。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 黑盒优化训练 | 使用行为算法生成的离线优化历史训练 RIBBO 模型 |
| 黑盒优化推理 | 加载权重，以优化历史为条件自回归生成查询点（Algorithm 1 + HRR） |
| 性能评估 | 计算 cumulative regret 与归一化目标值曲线 |
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
modelscope download --model OneScience/RIBBO --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem、bio、all（全领域）
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem（全领域）
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

（请在此处说明训练数据来源和获取方式：RIBBO 使用 BBOB 合成函数等基准，由 7 个行为算法（Random Search、Shuffled Grid Search、Hill Climbing、Regularized Evolution、Eagle Strategy、CMA-ES、GP-EI）生成离线优化历史，每条历史经 regret-to-go 增强。可通过 `scripts/generate_data.py` 生成。）

### 训练

```bash
python scripts/train.py --config conf/ribbo_config.yaml --seed 0 --function rastrigin
```

训练会在 `checkpoints/` 下保存 `ribbo_rastrigin_seed0_final.pt`。

### 训练权重

| 权重文件 | 说明 |
| :--- | :--- |
| `weight/ribbo_rastrigin_seed0_final.pt` | RIBBO 模型（GPT，d=10 BBOB Rastrigin，20000 steps 训练） |

### 推理

```bash
python scripts/infer.py --config conf/ribbo_config.yaml --seed 0 --function rastrigin --checkpoint weight/ribbo_rastrigin_seed0_final.pt
```

推理结果保存至 `outputs/infer_rastrigin_seed0.json`。

### 评估和可视化

```bash
python scripts/evaluate.py --config conf/ribbo_config.yaml --seed 0 --function rastrigin
```

评估输出 cumulative regret 与归一化目标值曲线至 `outputs/eval_rastrigin_seed0.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为 RIBBO 原始论文的复现版本：Reinforced In-Context Black-Box Optimization (arXiv:2402.17423)。
- Song, L., Gao, C.-X., Xue, K., Wu, C., Li, D., Hao, J., Zhang, Z., Qian, C. (2024). Reinforced In-Context Black-Box Optimization. arXiv:2402.17423.

