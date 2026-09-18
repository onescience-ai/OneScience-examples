<p align="center">
  <strong>
    <span style="font-size: 30px;">GP_for_TO</span>
  </strong>
</p>

# 模型介绍

GP_for_TO（Physics-informed GP-TO）是美国西北大学相关团队提出的物理信息高斯过程拓扑优化框架，可在无须显式网格离散的情况下，对复杂设计域中的材料分布与物理状态变量进行协同优化。  

论文：Simultaneous and Meshfree Topology Optimization with Physics-informed Gaussian Processes
https://arxiv.org/abs/2408.03490

# 模型描述

GP_for_TO 基于深度神经网络参数化均值函数的高斯过程架构，面向 Stokes 流耗散功最小化等流体拓扑优化问题开展无网格求解。

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
modelscope download --model OneScience/GP_for_TO --local_dir ./GP_for_TO
cd GP_for_TO
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
该方法不依赖预生成的拓扑结构或物理场标签，而是在设计域内采样空间点，利用 Stokes 流控制方程、边界条件、耗散功目标及材料体积分数约束构造物理信息训练信号；论文选取 Rugby、Pipe Bend、Diffuser 和 Double Pipe 四类流体拓扑优化算例进行测试，并使用 COMSOL 的 SIMP 求解结果进行对比验证。

### 训练

默认训练配置保留原始设置：`N_col_domain=10000`、`N_train_per_BC=25`、`num_iter=50000`、`diff_method=Numerical`。

```bash
python scripts/train.py
```
### 训练权重
本仓库在weights/文件夹内提供GP_for_TO模型权重，该权重即将上传。

## 推理

```bash
python scripts/inference.py
```

默认读取 `weight/gp_for_to.pt`，输出：

```text
result/inference/predictions.npz
result/inference/inference_summary.json
```

`predictions.npz` 中包含 `x`、`u`、`v`、`p`、`ro` 五组数组。

## 预测结果可视化

```bash
python scripts/result.py
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- GP_for_TO 原始论文：[Simultaneous and Meshfree Topology Optimization with Physics-informed Gaussian Processes](https://arxiv.org/abs/2408.03490)。
- 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
