<p align="center">
  <strong>
    <span style="font-size: 30px;">Transolver-Airfoil-Design</span>
  </strong>
</p>

# 模型介绍

Transolver-Airfoil-Design 是基于清华大学 THUML 团队提出的 Transolver 构建的二维翼型外流场预测模型，可对翼型外流场分布及气动性能进行快速预测。

论文：[Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)

# 模型描述
Transolver-Airfoil-Design 基于引入 Physics-Attention 的 Transformer 架构，适配 AirfRANS 非结构网格翼型数据训练，面向翼型开展速度场、压力场及阻力系数预测。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 翼型气动设计 | 快速预测二维翼型外流场，用于候选外形筛选 |
| 工业仿真加速 | 面向复杂几何 PDE 代理求解和大规模仿真加速 |

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
modelscope download --model OneScience/Transolver-Airfoil-Design --local_dir ./Transolver-Airfoil-Design
cd Transolver-Airfoil-Design
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
OneScience 社区提供可供训练的 `AirfRANS`，可通过下述命令下载，并确认 `config/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/airfrans --local_dir ./data
```

### 训练

```bash
python scripts/train.py
```

默认训练会保存 checkpoint：

```text
./weight/best.pth
```

### 训练权重
本仓库在weights/文件夹内提供基于 AirfRANS 数据预训练的模型权重， 可用于直接推理。

### 推理

```bash
python scripts/inference.py
```

### 评估和可视化

```bash
python scripts/result.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Transolver 原始论文：[Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。