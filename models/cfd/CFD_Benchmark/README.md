<p align="center">
  <strong>
    <span style="font-size: 30px;">CFD_Benchmark</span>
  </strong>
</p>

# 模型介绍
CFD_Benchmark 是一个面向神经偏微分方程（PDE）求解器研究的开源深度学习基准库，基于清华大学开源项目 Neural-Solver-Library 扩展而来，在原有神经算子与物理场建模框架基础上支持 DDP 并行训练，并引入了新的模型和数据集。它适合神经 PDE 求解器评测、CFD 场景下的深度学习建模、多模型性能对比、大规模并行训练实验、物理仿真数据集构建与算法基准测试等场景。

本库目前支持以下基准测试：

- 来自 [[FNO]](https://arxiv.org/abs/2010.08895) 和 [[geo-FNO]](https://arxiv.org/abs/2207.05209) 的六个标准基准
- PDEBench [[NeurIPS 2022 Track 数据集与基准]](https://arxiv.org/abs/2210.07182)，用于自回归任务的基准测试
- ShapeNet-Car 数据集 [[TOG 2018]](https://dl.acm.org/doi/abs/10.1145/3197517.3201325)，用于工业设计任务的基准测试
- BubbleML 数据集[[Multiphase Multiphysics Dataset]](https://arxiv.org/abs/2307.14623),用于研究多物理相变现象

---

支持的神经求解器

以下是支持的神经 PDE 求解器列表：


- **Transolver** - Transolver: A Fast Transformer Solver for PDEs on General Geometries [[ICML 2024]](https://arxiv.org/abs/2402.02366) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transolver.py)
- **ONO** - Improved Operator Learning by Orthogonal Attention [[ICML 2024]](https://arxiv.org/abs/2310.12487v3) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/ONO.py)
- **Factformer** - Scalable Transformer for PDE Surrogate Modeling [[NeurIPS 2023]](https://arxiv.org/abs/2305.17560) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Factformer.py)
- **U-NO** - U-NO: U-shaped Neural Operators [[TMLR 2023]](https://openreview.net/pdf?id=j3oQF9coJd) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_NO.py)
- **LSM** - Solving High-Dimensional PDEs with Latent Spectral Models [[ICML 2023]](https://arxiv.org/pdf/2301.12664) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/LSM.py)
- **GNOT** - GNOT: A General Neural Operator Transformer for Operator Learning [[ICML 2023]](https://arxiv.org/abs/2302.14376) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/GNOT.py)
- **F-FNO** - Factorized Fourier Neural Operators [[ICLR 2023]](https://arxiv.org/abs/2111.13802) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/F_FNO.py)
- **U-FNO** - An enhanced Fourier neural operator-based deep-learning model for multiphase flow [[Advances in Water Resources 2022]](https://www.sciencedirect.com/science/article/pii/S0309170822000562) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_FNO.py)
- **Galerkin Transformer** - Choose a Transformer: Fourier or Galerkin [[NeurIPS 2021]](https://arxiv.org/abs/2105.14995) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Galerkin_Transformer.py)
- **MWT** - Multiwavelet-based Operator Learning for Differential Equations [[NeurIPS 2021]](https://openreview.net/forum?id=LZDiWaC9CGL) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/MWT.py)
- **FNO** - Fourier Neural Operator for Parametric Partial Differential Equations [[ICLR 2021]](https://arxiv.org/pdf/2010.08895) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/FNO.py)
- **Transformer** - Attention Is All You Need [[NeurIPS 2017]](https://arxiv.org/pdf/1706.03762) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Transformer.py)

- **GFNO** - Group Equivariant Fourier Neural Operators for Partial Differential Equations[[2023 Poster]](https://arxiv.org/pdf/1706.03762)[[Code]](https://github.com/divelab/AIRS/blob/main/OpenPDE/G-FNO/models/GFNO.py)

部分视觉网络也可作为结构化几何任务的良好基线：

- **Swin Transformer** - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[ICCV 2021]](https://arxiv.org/abs/2103.14030) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Swin_Transformer.py)
- **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation [[MICCAI 2015]](https://arxiv.org/pdf/1505.04597) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/U_Net.py)

一些经典几何深度模型也被包含用于设计任务：

- **Graph-UNet** - Graph U-Nets [[ICML 2019]](https://arxiv.org/pdf/1905.05178) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/Graph_UNet.py)
- **GraphSAGE** - Inductive Representation Learning on Large Graphs [[NeurIPS 2017]](https://arxiv.org/pdf/1706.02216) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/GraphSAGE.py)
- **PointNet** - PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [[CVPR 2017]](https://arxiv.org/pdf/1612.00593) [[Code]](https://github.com/thuml/Neural-Solver-Library/blob/main/models/PointNet.py)

还包含图神经网络：

- **MeshGraphNet** LEARNING MESH-BASED SIMULATION WITH GRAPH NETWORKS[ICLR 2021](https://arxiv.org/abs/2010.03409) [[Code]](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)



## 适用场景

| 场景 | 说明 |
|---|---|
| 神经 PDE 求解器评测 | 在统一流程下对 FNO、Transolver、GNOT、ONO、U-NO 等模型进行训练、推理和性能对比 |
| 自回归物理预测 | 基于 PDEBench 等数据集逐步预测 PDE 状态随时间的演化 |
| 多物理场建模 | 基于 BubbleML 等数据集研究多相流、多物理耦合和相变现象 |
| ModelScope/OneCode 运行	| 作为独立模型包下载后直接安装依赖并运行脚本 |


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
modelscope download --model OneScience/CFD_Benchmark --local_dir ./CFD_Benchmark 
cd CFD_Benchmark 
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

请参考上文基准测试说明中的数据集下载链接，下载所需数据集。

来自 [[FNO]](https://arxiv.org/abs/2010.08895) 和 [[geo-FNO]](https://arxiv.org/abs/2207.05209) 的六个标准基准数据集，可以通过[此链接](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8)下载。

PDEBench [[NeurIPS 2022 Track 数据集与基准]](https://arxiv.org/abs/2210.07182)用于自回归任务的基准测试数据集，可以通过[此链接](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)下载。

ShapeNet-Car [[TOG 2018]](https://dl.acm.org/doi/abs/10.1145/3197517.3201325)，用于工业设计任务的基准测试数据集，可以通过[[此链接]](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip)下载。

BubbleML [[Multiphase Multiphysics Dataset]](https://arxiv.org/abs/2307.14623)用于研究多物理相变现象数据集，可以通过[[此链接]](https://github.com/HPCForge/BubbleML/blob/main/bubbleml_data/README.md)下载。

OneScience 社区也提供可供训练的cfd_benchmark数据集，用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确：
```
modelscope download --dataset OneScience/cfd_benchmark  --local_dir ./data
```
### 训练

```bash
python scripts/train.py
```

### 训练权重
本仓库在weights/文件夹内提供基于基准OneScience/cfd_benchmark数据集训练的权重，即将上传该权重。

### 推理

```bash
python scripts/inference.py
```

推理会读取 `paths.weight_path` 指向的训练权重，并将指标写入：

```text
./results/{train.save_name}/metrics.json
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证
- 参考仓库：[Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

