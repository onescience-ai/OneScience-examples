<p align="center">
  <strong>
    <span style="font-size: 30px;">UPT</span>
  </strong>
</p>

# 模型介绍
UPT（Universal Physics Transformer）是由 Benedikt Alkin 等人提出的通用物理 Transformer 框架，通过将不规则网格或粒子数据编码到紧凑的潜在表示中，并在潜在空间内传播信息，实现灵活且可扩展的物理代理建模。

本仓库由 OneScience 技能流程依据论文描述与官方配置，独立复现 UPT 在 ShapeNetCar 数据集上的无 SDF 实验。模型根据汽车表面的三维点坐标预测对应位置的稳态表面压力，不使用 SDF、法向量、网格连接关系或流场变量作为模型输入。

论文：[Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators](https://proceedings.neurips.cc/paper\_files/paper/2024/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf)

# 模型描述
UPT 是面向物理代理建模的 Transformer 架构。对于本仓库实现的 ShapeNetCar 任务，模型输入汽车表面的三维点坐标 \((x,y,z)\)，并在指定的表面查询坐标处输出对应的标量压力值。

模型采用“坐标编码—Perceiver 编码器—潜在空间 Transformer—Perceiver 解码器—压力投影”结构：首先对缩放后的三维坐标进行连续正弦余弦位置编码，并通过 MLP 生成输入特征；随后由 Perceiver 交叉注意力将可变数量的输入点压缩为一组可学习的潜在 token，再通过多层 Transformer Block 在潜在空间中建模全局几何关系；解码阶段将查询坐标编码为查询 token，通过交叉注意力读取潜在表示，最后经归一化和线性投影得到各查询点的表面压力预测。


## 适用场景

| 场景 | 说明 |
| --- | --- |
| 汽车表面压力预测 | 根据 ShapeNetCar 汽车表面的三维点坐标预测稳态气动压力分布 |
| 不规则点集建模 | 对不依赖规则网格和显式网格连接关系的三维表面点集进行物理量回归 |
| 科学计算代理模型 | 在固定数据分布和流动条件下近似高成本数值模拟，实现快速批量预测 |
| 可变采样点推理 | 支持可变数量的输入点，并在模型训练分布内对指定表面坐标查询压力值 |


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
modelscope download --model OneScience/UPT --local_dir ./UPT
cd UPT
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

OneScience 社区在 魔搭上 提供 [`OneScience/ShapeNetCar`](https://modelscope.cn/datasets/OneScience/ShapeNetCar) 数据集，可通过以下命令下载：

```bash
modelscope download --dataset OneScience/ShapeNetCar --local_dir ./data
```

请确保 `config/config.yaml` 中的 `data.root` 指向 `preprocessed_data` 目录。数据集共包含 889 个汽车几何样本，代码按照固定随机种子划分为 700 个训练样本和 189 个测试样本。每个预处理样本的主要文件包括：

- `pos.npy`：所有节点的三维坐标，形状为 `[num_nodes, 3]`；
- `surf.npy`：表面节点掩码，形状为 `[num_nodes]`；
- `y.npy`：目标物理场，形状为 `[num_nodes, 4]`，前三个通道为速度分量，第 4 个通道为压力；
- `x.npy`：节点特征，形状为 `[num_nodes, 7]`，本实现仅使用其中的法向量辅助筛选汽车表面节点，不将其输入模型；
- `edge_index.npy`：图连接关系，本实现不使用该文件作为模型输入。

数据加载阶段会筛选汽车网格表面的 3586 个有效点，对三维坐标进行区间缩放，并使用配置中的均值和标准差归一化压力。训练时从每个样本中随机选择 80%～100% 的表面点作为编码器输入，同时在全部表面点上计算压力预测损失；测试时使用全部有效表面点作为输入和查询点。

### 训练

默认配置 `config/config.yaml` 为 UPT ShapeNetCar 无 SDF 复现实验配置。

```bash
python scripts/train.py --config config/config.yaml
```

训练采用归一化表面压力的掩码均方误差作为损失。每个 epoch 的训练与测试指标会写入：

```text
results/train_metrics.jsonl
```

脚本按照配置的间隔将批次损失、学习率、归一化压力 MSE、归一化压力 MSE × 100 和原始压力 RMSE 输出至终端。测试集归一化压力 MSE × 100 最低的 checkpoint 保存为：

```text
weight/best_model.pth
```

### 训练权重

本仓库在 `weight/` 文件夹内提供基于 ShapeNetCar 数据训练的 UPT 模型权重，可直接加载用于表面压力推理和数值评估。

### 推理评估可视化

以下命令在固定的 189 个测试样本上执行推理，并逐样本打印归一化压力 MSE × 100 和原始压力 RMSE：

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/best_model.pth
```

### 评估和可视化

运行前请确认 `config/config.yaml` 中的 `data.root` 路径有效，且 `weight/best_model.pth` 已存在。

数值评估由 `scripts/inference.py` 完成。脚本加载最佳 checkpoint，在固定测试集上计算归一化压力 MSE、归一化压力 MSE × 100、归一化压力 MAE、原始压力 MSE 和原始压力 RMSE，同时保存每个样本的预测结果：

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/best_model.pth \
  --device auto
```

推理结果默认保存至：

```text
results/inference_metrics.json
results/predictions/
└── <sample_id>.npz
```

其中，`inference_metrics.json` 包含测试集聚合指标、论文参考指标以及各样本的评估结果；每个 `<sample_id>.npz` 包含表面点坐标、归一化预测与真实压力、原始尺度预测与真实压力以及绝对误差。

完成推理后，运行独立可视化脚本：

```bash
python scripts/result.py --config config/config.yaml
```

可视化结果默认保存至：

```text
results/figures/
├── training_metrics.png
├── <sample_id>_pressure_comparison.png
└── visualization_metadata.json
```

其中：

- `training_metrics.png`：训练归一化 MSE 和测试归一化 MSE × 100 曲线，并标注论文参考值；
- `<sample_id>_pressure_comparison.png`：真实表面压力、UPT 预测表面压力及绝对压力误差的三维对比；
- `visualization_metadata.json`：记录预测数据来源、评估文件来源、可视化文件路径及绘图参数。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators](https://proceedings.neurips.cc/paper_files/paper/2024/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf)
- 官方实现：[ml-jku/UPT](https://github.com/ml-jku/UPT)，采用 [MIT License](https://github.com/ml-jku/UPT/blob/main/LICENSE.md)。
- 本仓库为依据论文描述与官方配置完成的独立复现，不代表论文作者或官方实现。论文、官方代码、ShapeNetCar 数据集及其他第三方资源分别受其原始版权声明、许可证和使用条款约束。

