<p align="center">
  <strong>
    <span style="font-size: 30px;">CNO</span>
  </strong>
</p>

# 模型介绍
CNO（Convolutional Neural Operator，卷积神经算子）是 **Bogdan Raonić 等人**提出的一种面向 **偏微分方程（PDE）算子学习**的神经算子模型，并发表于 **NeurIPS 2023**。CNO 将卷积神经网络与连续函数空间中的算子学习相结合，通过抗混叠激活、滤波升降采样等设计降低离散化和分辨率变化带来的误差，可用于从 PDE 的初始条件、源项或参数场直接预测对应的 PDE 解。本项目基于 OneScience 技能，独立复现了 CNO 论文中二维不可压缩 Navier–Stokes 方程水平速度分量从 \(t=0\) 到 \(T=1\) 的预测实验。

论文：[Convolutional Neural Operators for Robust and Accurate Learning of PDEs](https://arxiv.org/abs/2302.01178)


# 模型描述
CNO 采用类似 **U-Net 的多尺度编码器—解码器结构**，结合局部卷积、跳跃连接以及带滤波的升降采样实现 PDE 的函数到函数映射。其核心是在非线性激活与尺度变换中引入 **抗混叠（anti-aliasing）机制**，减小离散化和分辨率变化引起的误差，使模型更好地保持连续算子的性质，从而提升 PDE 解算子学习的精度与跨分辨率鲁棒性。




## 适用场景

| 场景        | 说明                                                |
| --------- | ------------------------------------------------- |
| PDE 解算子学习 | 学习初始条件、源项或参数场到 PDE 解之间的函数映射                       |
| 流体动力学预测   | 可用于 Navier–Stokes、Compressible Euler 等流体问题的快速代理预测 |
| 多尺度物理场建模  | 适用于同时包含低频和高频空间结构的复杂 PDE 解                         |
| 跨分辨率预测    | 可在不同空间离散分辨率下进行推理，用于评估模型的分辨率泛化能力                   |
| ID/OOD 泛化 | 支持在分布内和分布外 PDE 参数或物理条件下评估模型鲁棒性                    |



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
modelscope download --model OneScience/CNO --local_dir ./CNO
cd CNO
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

实验使用 RPB_CNO 数据集中的二维不可压缩 Navier–Stokes 数据。训练文件为
`NavierStokes_64x64_IN.h5`，每个样本包含：

- `input`：\(t=0\) 时刻的水平速度分量，shape 为 `(64, 64)`；
- `output`：\(T=1\) 时刻的水平速度分量，shape 为 `(64, 64)`。

加载后会增加通道维度，训练批次的输入和标签 shape 均为
`(batch_size, 1, 64, 64)`。

当前配置使用 750 个训练样本、128 个验证样本和 128 个分布内测试样本。
分布外测试数据来自 `NavierStokes_128x128_OUT.h5`，共 128 个样本。

可通过下述命令下载数据：

```bash
modelscope download --dataset OneScience/RPB_CNO --local_dir ./data
```
下载后，将 config/config.yaml 中的 paths.data_dir 指向包含 HDF5
数据文件的目录

### 训练

默认配置对应论文中的二维不可压缩 Navier–Stokes 实验：谱黏性约为\(\nu=4\times10^{-4}\)，模型学习从 \(t=0\) 初始水平速度分量到\(T=1\) 速度分量的映射。

```bash
python scripts/train.py --config config/config.yaml --device auto
```

验证集物理空间相对中位 L1 最低时的完整训练状态保存至 weight/best_model.pth，其中包含模型、优化器、学习率调度器和归一化参数。



### 训练权重

运行download.sh脚本，可下载复现训练得到的最优权重`weight/best_model.pth`，可直接用于推理微调；

### 推理

运行前需确保配置中的数据路径有效，且 `weight/best_model.pth` 已存在。模型执行从 \(t=0\) 初始水平速度分量到 \(T=1\) 速度分量的单步预测，
不包含多步闭环或轨迹滚动预测。默认推理 batch size 为 16。

```bash
python scripts/inference.py --config config/config.yaml 
```

### 评估和可视化

完成训练和推理后运行：

```bash
python scripts/result.py --config config/config.yaml --sample-index 0
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 原论文链接：[Convolutional Neural Operators for Robust and Accurate Learning of PDEs](https://arxiv.org/abs/2302.01178)
- 本项目为 CNO 论文的独立复现。官方实现代码采用 MIT License；本项目代码、模型权重、训练数据及第三方依赖分别适用其各自的许可条款
