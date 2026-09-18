<p align="center">
  <strong>
    <span style="font-size: 30px;">PINNsformer</span>
  </strong>
</p>

# 模型介绍

PINNsFormer 是由美国佐治亚理工学院与卡内基梅隆大学研究团队提出的基于 Transformer 的物理信息神经网络框架，可对具有时间依赖性的偏微分方程解及其对应物理场进行快速预测。

论文：[PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks](https://arxiv.org/abs/2307.11833)。

# 模型描述

PINNsFormer 基于 Transformer 编码器—解码器和多头注意力架构，面向对流方程、反应方程、波动方程及 Navier–Stokes 方程等时变偏微分方程开展数值求解。



# 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 时变 PDE 求解 | 通过物理残差、边界条件和初始条件约束训练连续场代理模型 |
| 物理信息神经网络验证 | 快速检查 PINNsformer 网络、损失函数、权重保存和推理链路 |
| 一维反应方程示例 | 基于解析解生成目标场，用于流程连通性验证和误差计算 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

* CPU 可用于小配置流程验证。
* 推荐使用 GPU 或 DCU 进行较大网格、较多 epoch 的训练。
* DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/PINNsformer --local_dir ./PINNsformer
cd PINNsformer
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

如需使用真实数据，可通过如下链接下载，并将 `conf/config.yaml` 中的 `data.data_dir` 进行正确的路径配置。

| 下载源 | 链接 | 提取码 | 下载后放置位置 |
|---|---|---|---|
| 百度网盘 | https://pan.baidu.com/s/1pM4ICc6FJX5pLF7WEoozxQ?pwd=5gha | `5gha` | `convection/convection.mat` 和 `navier_stokes/cylinder_nektar_wake.mat` |

### 训练

```bash
python scripts/train.py
```
默认配置使用较小网格和较少 LBFGS 迭代，便于快速跑通流程。如需恢复原始示例规模，可修改 `conf/config.yaml`：

```yaml
data:
  x_num: 101
  t_num: 101

training:
  epochs: 500
```

### 训练权重
本仓库在weights/文件夹内提供基于pinnsformer数据训练的模型权重，该权重即将上传。

### 推理

```bash
python scripts/inference.py
```

推理会读取 `weight/1dreaction_pinnsformer.pt`，并保存：

```text
result/prediction.npz
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

* PINNsformer 原始论文：[PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks](https://arxiv.org/abs/2307.11833)。
* 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
