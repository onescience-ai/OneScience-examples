<p align="center">
  <strong>
    <span style="font-size: 30px;">RNO</span>
  </strong>
</p>

# 模型介绍
RNO（Radon Neural Operator）是由浙江工业大学团队提出的偏微分方程神经算子，通过 Radon 变换在正弦图域中学习兼具全局与局部特征的 PDE 解算映射。

本仓库由 OneScience 技能流程依据论文描述独立复现在Darcy Flow 数据集上的实验，使用 RNO 学习 Darcy Flow 的参数化解算子，根据多孔介质的渗透率/扩散系数场预测对应的稳态压力解。

论文：[Solving Partial Differential Equations via Radon Neural Operator](https://proceedings.neurips.cc/paper\_files/paper/2025/file/e66233a208ef32f56df6312263239fa0-Paper-Conference.pdf)

# 模型描述
RNO 是面向参数化偏微分方程求解的神经算子模型。对于 Darcy Flow 任务，模型输入二维渗透率/扩散系数场 \(a(x,y)\)，输出对应的标量压力解 \(u(x,y)\)。

模型采用“特征提升—Physics-Attention—Radon Block—输出投影”结构：首先将输入场与空间坐标编码为高维特征，再通过 Physics-Attention 提取非局部信息；Radon Block 将特征投影至正弦图域，利用角度重加权和正弦图卷积学习不同投影方向的贡献，并通过滤波反投影恢复空间特征，最终输出 Darcy 方程的预测解。


## 适用场景

| 场景 | 说明 |
| --- | --- |
| Darcy 渗流预测 | 根据二维渗透率或扩散系数场预测多孔介质中的稳态压力分布 |
| 参数化 PDE 求解 | 学习输入系数、初始条件或边界条件到 PDE 解之间的算子映射 |
| 科学计算代理模型 | 替代部分高成本数值求解过程，实现快速批量预测 |
| 跨分辨率预测 | 利用神经算子的离散化不变性，在不同空间分辨率上执行推理 |


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
modelscope download --model OneScience/RNO --local_dir ./RNO
cd RNO
```

### 安装运行环境


**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

OneScience 社区在 ModelScope 的 [`OneScience/cfd_benchmark`](https://modelscope.cn/datasets/OneScience/cfd_benchmark) 数据集中提供 Darcy Flow 数据，可通过以下命令下载：

```bash
modelscope download \
  --dataset OneScience/cfd_benchmark \
  --local_dir ./data
```
Darcy Flow 数据位于下载目录的 data/darcy/ 下：
```
data/data/darcy/
├── piececonst_r421_N1024_smooth1.mat
└── piececonst_r421_N1024_smooth2.mat
```
请确保 config.yaml 中的 data.root 指向Darcy Flow 数据路径，两个 MAT 文件均包含 1024 组规则网格样本，原始空间分辨率为 421 × 421，主要字段包括：
- coeff：Darcy 方程的渗透率/扩散系数场；
- sol：对应的稳态压力解。
本实验使用 piececonst_r421_N1024_smooth1.mat 构建训练集，该数据是规则网格上的二维标量场，使用 piececonst_r421_N1024_smooth2.mat 构建测试集，并按照实验配置对原始场进行下采样和归一化处理。

### 训练

默认配置 `config/config.yaml` RNO Darcy Flow 复现实验配置。

```bash
python scripts/train.py --config config/config.yaml
```

训练过程中每个 epoch 的指标都会写入日志，并按照配置的间隔输出至终端；训练 Relative L2 最低的 checkpoint 保存为：

```text
weight/best_model.pth
```



### 训练权重

本仓库在`weight/`文件夹内提供基于 Darcy Flow 数据预训练的 RNO 模型权重，可直接加载推理、预训练。

### 推理评估可视化

以下命令在固定测试集上执行推理并实时打印平均逐样本 Relative L2：

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/best_model.pth \
```

### 评估和可视化

运行前请确认 `config/config.yaml` 中的数据路径有效，且 `weight/best_model.pth` 已存在。
数值评估由 `scripts/inference.py` 完成。脚本加载最佳 checkpoint，在固定测试集上计算 Relative L2、Gradient Relative L2 及其与论文参考结果的差异，同时保存预测数据：

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/best_model.pth \
  --device auto
```
推理结果默认保存至：
```
results/evaluation_metrics.json
results/predictions.npz
```
其中 predictions.npz 包含预测压力场、真实压力场、归一化后的渗透率场及各样本的 Relative L2。
完成推理后，运行独立可视化脚本：
```
python scripts/result.py --results results
```
可视化结果保存至：
```
results/training_curve.png
results/darcy_prediction.png
results/visualization_summary.json
```
其中：
- training_curve.png：训练总 Loss、Relative L2 和 Gradient Relative L2 曲线；
- darcy_prediction.png：渗透率场、真实压力场、预测压力场及绝对误差对比；
- visualization_summary.json：可视化文件路径、测试 Relative L2 和论文对比结论。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[Solving Partial Differential Equations via Radon Neural Operator](https://proceedings.neurips.cc/paper_files/paper/2025/file/e66233a208ef32f56df6312263239fa0-Paper-Conference.pdf)
- 官方实现：[wenbin-lu/Radon-Neural-Operator](https://github.com/wenbin-lu/Radon-Neural-Operator)，采用 [MIT License](https://github.com/wenbin-lu/Radon-Neural-Operator/blob/main/LICENSE)。
- 本仓库为依据论文描述完成的独立复现，不代表论文作者或官方实现。论文、官方代码、数据集及其他第三方资源分别受其原始版权声明、许可证和使用条款约束。

