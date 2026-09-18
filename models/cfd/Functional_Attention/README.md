<p align="center">
  <strong>
    <span style="font-size: 30px;">Functional Attention</span>
  </strong>
</p>

# 模型介绍

Functional Attention 是慕尼黑工业大学相关团队提出的分辨率无关算子学习框架，可对偏微分方程解、气动场及三维点云分割结果等连续函数进行快速预测。

论文：[Functional Attention: From Pairwise Affinities to Functional Correspondences](https://arxiv.org/abs/2605.31559)

# 模型描述

Functional Attention 基于自适应基函数与函数映射机制，使用AirfRANS 气动数据训练与评测，面向跨离散方式和跨分辨率的 PDE 求解、三维分割与回归任务。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 二维翼型流场代理建模 | 在 AirfRANS 非结构网格上预测速度、压力和湍流黏度等点级物理场 |
| OOD Reynolds 泛化评估 | 使用 `reynolds_train -> reynolds_test` 验证跨雷诺数分布外泛化 |
| CFD 数值求解器代理加速 | 以神经算子近似 RANS 仿真结果，用于快速场预测和设计筛选 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU、DCU 或 HCU 运行完整训练。
- CPU 可用于导入检查和极小规模代码连通性验证。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/Functional_Attention --local_dir ./Functional_Attention
cd Functional_Attention
```


### 安装运行环境

**DCU环境**

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```
**GPU环境**

```
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍
OneScience社区提供可供训练的 AirfRANS 数据，用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确：
```
modelscope download --dataset OneScience/airfrans --local_dir ./data
```

用户也可通过如下链接下载原始数据：

```text
https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip
```

### 训练

默认执行 Reynolds OOD 实验：

```bash
python scripts/train.py \
  --config config/config.yaml \
  --task reynolds
```

### 训练权重
训练过程中，验证集 `Lv + Ls` 最优的 checkpoint 保存至：

```text
weight/best_model.pth
```

该 checkpoint 包含模型参数、优化器、学习率调度器、训练 epoch、实验配置及归一化统计量。

### 推理

使用当前最佳 checkpoint 推理 3 个 Reynolds OOD 测试样本：

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/best_model.pth \
  --task reynolds \
  --max-cases 3
```
指标包括四个物理场的 relative L2、表面压力 relative L2，以及明确标注的 `pressure_only_*` 升阻力系数指标。


### 结果可视化

`result.py` 读取训练历史和推理产生的 `.npz` 文件，生成训练曲线及真值、预测、绝对误差云图：

```bash
python scripts/result.py \
  --config config/config.yaml \
  --task reynolds
```

可视化结果默认保存至：

```text
results/figures/
results/visualization_manifest_reynolds.json
```
# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Functional Attention 原始论文：[Functional Attention: From Pairwise Affinities to Functional Correspondences](https://arxiv.org/abs/2605.31559)
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理；公开分发前请根据上游项目和数据集确认许可证要求。

