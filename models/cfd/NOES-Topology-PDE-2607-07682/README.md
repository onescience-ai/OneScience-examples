<p align="center">
  <strong>
    <span style="font-size: 30px;">NOES-Topology-PDE-2607-07682</span>
  </strong>
</p>

# 模型介绍

NOES-Topology-PDE-2607-07682 是论文复现任务 `2607.07682` 的本地 Tier 1 产物，用 PCA 潜空间和 DeepONet 解码器学习光子拓扑设计的 PDE 响应。当前包面向接口、训练和诊断验证；源规格明确指出真实 Maxwell/RCWA 材料、边界、谐波和标签仍未提供。

# 模型描述

- `model/deeponet.py`：DeepONet branch/trunk 网络和解码器。
- `model/pca_model.py`、`model/pca_pipeline.py`：PCA 潜变量建模与数据管线。
- `model/dataset.py`：训练和验证数据集封装。
- `model/binarize.py`：设计二值化工具。
- 其余 `model/` 文件提供 PDE、损失、指标、搜索和优化相关组件。
- 已提供 `weight/best_model.pt`，对应本地 photonics Tier 1 训练运行的最佳检查点。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用 `scripts/train.py` 和 `conf/photonics.yaml` 训练 DeepONet/PCA 代理模型。 |
| 模型推理 | 当前包未发现独立推理或预测入口脚本。 |
| 评估和诊断 | 使用 `scripts/evaluate.py` 计算验证 MSE 和二值化率。 |
| 本地接口验证 | 使用实际存在的 `scripts/smoke.py` 或 `scripts/smoke_test.py` 执行轻量检查。 |

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
modelscope download --model OneScience/NOES-Topology-PDE-2607-07682 --local_dir ./NOES-Topology-PDE-2607-07682
cd NOES-Topology-PDE-2607-07682
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

训练入口使用本地复现管线生成的合成 photonics 数据；源目录没有可发布的数据集文件或明确的 ModelScope 数据集仓库。

### 训练

从模型包根目录运行：

```bash
PYTHONPATH=model python scripts/train.py --config conf/photonics.yaml --checkpoint-dir ./weight/runtime
```

### 训练权重

已提供 `weight/best_model.pt`。它是 Tier 1 本地训练检查点，包含模型状态、训练配置、PCA 状态、坐标和最佳验证 MSE。

### 推理

<!-- 推理脚本未在 scripts/ 目录中找到，请补充 -->

### 评估和可视化

使用已有检查点评估：

```bash
PYTHONPATH=model python scripts/evaluate.py --config conf/photonics.yaml --checkpoint weight/best_model.pt --out ./evaluation/report.json
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

论文来源：`2607.07682`。源产物未包含完整论文标题、DOI 或可发布的引用链接，请以论文元数据补充正式 BibTeX 引用。

本模型包按 Apache License 2.0 元数据发布；真实物理求解器参数和数据标签的缺失限制了其作为论文最终结果的解释范围。

