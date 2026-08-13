<p align="center">
  <strong>
    <span style="font-size: 30px;">PointNetCFD</span>
  </strong>
</p>

# 模型介绍
PointNetCFD 是由 Ali Kashefi、Davis Rempe 和 Leonidas J. Guibas 提出的点云流场预测模型。该方法将不规则几何区域中的非结构 CFD 网格节点直接表示为点云，并利用 PointNet 编码几何信息和空间位置，逐点预测两个速度分量和压力。本仓库由 OneScience 技能流程依据论文描述独立复现。

论文：[A Point-Cloud Deep Learning Framework for Prediction of Fluid Flow Fields on Irregular Geometries](https://arxiv.org/abs/2010.09469)

# 模型描述
PointNetCFD 是面向非结构网格的 CFD 点级回归模型。每个样本包含 1024 个点，以节点坐标 `(x, y)` 为输入，逐点预测 `(u, v, p)`。模型通过 T-Net 对输入和特征进行对齐，经共享 MLP 与全局最大池化提取局部和全局特征，融合后通过 `512 → 256 → 128 → 128 → 3` 解码器完成流场预测。坐标保持原始物理尺度，输出变量按训练集统计量归一化至 `[0, 1]`。



## 适用场景

| 场景         | 说明                                 |
| ---------- | ---------------------------------- |
| CFD 流场预测     | 根据非结构网格节点的二维空间坐标 (x, y)，逐点预测速度分量 (u, v) 和压力 p             |
| 不规则几何建模     | 直接利用点云表示不同物体边界和非结构网格，无需将 CFD 数据插值到规则网格               |
| 几何泛化   | 所对应的实验可测试未出现几何形状的预测能力 |


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
modelscope download --model OneScience/PointNetCFD --local_dir ./PointNetCFD
cd PointNetCFD
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
OneScience 社区提供可供训练的 PointNetCFD 数据，用户可通过下述命令下载，并确认 `config/config.yaml` 中的 `paths.data_dir` 指向下载后的数据目录：

```bash
modelscope download --dataset OneScience/pointnet_cfd --local_dir ./data
```
其中 CFDdata.npy 的单个样本可以表示为 1024 × 5 的点云数据矩阵：
`[x, y, p, u, v]`，同时提供训练集、验证集和测试集对应的索引文件。

### 训练

默认配置 `config/config.yaml` 对应论文主实验设置。

```bash
python scripts/train.py --config config/config.yaml
```

训练过程中会通过标准输出实时打印每个 epoch 的训练 loss、验证 loss 和评估指标。验证集 MSE 最优的 checkpoint 保存至：

```text
weight/best_model.pth
```

训练历史、实际生效配置和训练摘要统一保存在 `results/` 目录。仅验证环境连通性时，可使用独立输出路径的最小冒烟测试：

```bash
python scripts/train.py --smoke-test
```

### 训练权重

运行download.sh脚本，可下载复现训练得到的最优权重`weight/best_model.pth`，可直接用于推理微调；

### 推理

运行前请确认 `config/config.yaml` 中的数据路径有效，且 `weight/best_model.pth` 已存在。以下命令在固定测试集上执行推理并实时打印归一化 MSE、各物理变量的 RMSE 和相对 L2 误差：

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/best_model.pth \
  --device auto \
  --output-dir results
```

推理结果保存为：

- `results/test_metrics.json`：测试指标及论文参考指标；
- `results/predictions.npz`：坐标、预测值、真实值及样本索引。

### 评估和可视化

数值评估由 `scripts/inference.py` 在推理时完成。可视化脚本依赖推理生成的 `results/predictions.npz`，因此应先完成上述推理，再运行：

```bash
python scripts/result.py \
  --predictions results/predictions.npz \
  --output-dir results/figures \
  --num-cases 3
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[A Point-Cloud Deep Learning Framework for Prediction of Fluid Flow Fields on Irregular Geometries](https://arxiv.org/abs/2010.09469)，[DOI: 10.1063/5.0033376](https://doi.org/10.1063/5.0033376)

- 本仓库保留原始论文及官方实现的来源与版权信息，其中官方代码采用 MIT License；论文、数据集及其他相关资源仍分别受其各自版权声明与使用条款约束。
