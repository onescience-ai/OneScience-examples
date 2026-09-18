<p align="center"><strong><span style="font-size: 30px;">aot-pot-pde-2605-15793</span></strong></p>

# 模型介绍

A runnable AOT-POT-inspired implementation for `2605.15793`, targeting next-frame prediction for heterogeneous, time-dependent PDE solution operators. The reproduction uses the explicit `[B,128,128,T=10,C=4]` input contract and predicts `[B,128,128,C=4]`.

论文：AOT-POT: Adaptive Operator Transformation for Large-Scale PDE Pre-training  
https://arxiv.org/abs/2605.15793

# 模型描述

- `model/aot.py`: AOT stream aggregation and transformation components.
- `model/aot_pot.py`: AOT-POT model composition.
- `model/embedding.py`: coordinate-aware patch and temporal embedding.
- `model/fourier.py`: Fourier operator proxy.
- Architecture: coordinate-aware patch embedding, weighted temporal aggregation, four AOT streams, Sinkhorn normalization, Fourier mixing, gated readout, and next-state projection.

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用 `scripts/train.py` 运行训练或合成数据 smoke run。 |
| 模型推理 | 使用 `scripts/infer.py` 加载权重进行 next-frame prediction。 |
| 评估和可视化 | 使用 `scripts/evaluate.py` 计算评估指标。 |
| 数据预处理 | 使用 `scripts/preprocess.py` 准备数据接口。 |

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
modelscope download --model OneScience/aot-pot-pde-2605-15793 --local_dir ./model
cd model
```

### 安装运行环境

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

（请在此处说明训练数据来源和获取方式）

### 训练

```bash
python scripts/train.py
```

### 训练权重

- `weight/checkpoint.pt` (2.9 MB)

### 推理

```bash
python scripts/infer.py
```

### 评估和可视化

```bash
python scripts/evaluate.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Lv, Q., Wang, H., Hao, Z., Wu, W., Xu, X., Zhou, B., Wu, F., and Zhang, C. “AOT-POT: Adaptive Operator Transformation for Large-Scale PDE Pre-training.” arXiv:2605.15793, 2026.

