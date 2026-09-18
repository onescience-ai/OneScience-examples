<p align="center"><strong><span style="font-size: 30px;">GRACE-SEDA</span></strong></p>

# 模型介绍

GRACE-SEDA 是面向全球高分辨率总水储量异常的自监督数据同化模型。模型融合 GRACE 的大尺度精度和 WGHM 的高分辨率结构。

论文：Global high-resolution total water storage anomalies from self-supervised data assimilation using deep learning algorithms  
https://doi.org/10.1038/s44221-024-00194-w

# 模型描述

该模型由 ETH Zurich 等机构团队提出。模型使用 JPL GRACE、WGHM、GLDAS 水文变量和地理坐标训练。模型通过残差编码器-解码器与自监督双约束，适用于 0.5° 全球 TWSA 重建、不确定性估计和水量平衡分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| TWSA 降尺度 | 生成 0.5° 高分辨率水储量异常。 |
| 自监督同化 | 平衡 GRACE 大尺度值和 WGHM 空间结构。 |
| 不确定性估计 | 使用五模型深度集合。 |
| ModelScope/OneCode 运行 | 验证数据、训练、推理、水文指标和可视化。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装
```bash
modelscope download --model OneScience/GRACE-SEDA --local_dir ./GRACE-SEDA
cd GRACE-SEDA
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍
论文使用 GRACE、WGHM、降水、ET、三类径流和经纬度九特征，以 0.5°、`32×32` patch 训练。虚拟数据保持九通道、自监督损失和五成员集合，仅减少月份和样本数。结果仅用于工程验证。
```bash
python scripts/fake_data.py
```

### 训练
单卡训练可使用：
```bash
python scripts/train.py
```
多卡训练可使用：
```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py
```
训练联合优化 GRACE patch 均值误差和 WGHM 结构相似性，单卡与双进程 DDP 均已通过。训练结果保存到：
```text
result/checkpoints/grace_seda.pt
result/training/metrics.json
```

### 训练权重
本仓库不在 `weight/` 中内置权重。作者在 https://gitlab.ethz.ch/spacegeodesy_public/grace_seda 提供核心代码、训练模型和权重。

### 推理
```bash
python scripts/inference.py
```
推理恢复五个模型并输出集合均值和不确定性，集合 shape 为 `[5,20,32,32]`。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化
```bash
python scripts/result.py
```
评估计算 WGHM 相关、GRACE 聚合误差和平均不确定性，并生成空间误差图。评估结果保存到：
```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息
| 平台 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |
# 引用与许可证
本仓库为 GRACE-SEDA 公开规格的独立工程复现版本。

原始论文采用 CC BY 4.0 许可证；原始论文、官方代码、模型权重和相关数据仍应按照各自项目的许可证及使用条款使用。
