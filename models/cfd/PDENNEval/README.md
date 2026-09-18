<p align="center">
  <strong>
    <span style="font-size: 30px;">PDENNEval</span>
  </strong>
</p>

# 模型介绍
PDENNEval 是中山大学团队提出的神经网络偏微分方程求解方法综合评测基准，可对函数学习与算子学习方法在多类 PDE 任务上的求解性能进行系统比较。

论文：[PDENNEval: A Comprehensive Evaluation of Neural Network Methods for Solving PDEs](https://www.ijcai.org/proceedings/2024/0573.pdf)

# 模型描述
PDENNEval 基于统一的神经网络 PDE 求解评测框架，使用高精度传统科学计算方法生成的 16 组数据，覆盖流体力学、材料科学、金融和电磁学等领域的 19 类 PDE 问题，面向 12 种神经网络方法开展求解精度、计算效率与鲁棒性评估。

## 适用场景

| 场景 | 说明 |
| --- | --- |
| PDE 神经算子训练验证 | 使用 FNO 在 PDEBench 风格 HDF5 数据上完成最小训练和推理闭环 |
| 神经网络 PDE 方法扩展 | `model/` 已包含 FNO、DeepONet、PINO-FNO、UNO、MPNN、UNet 等模型定义 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |
| 快速连通性检查 | 使用 `fake_data.py` 和 `smoke_models.py` 验证数据、模型和脚本可运行 |

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
modelscope download --model OneScience/PDENNEval --local_dir ./PDENNEval
cd PDENNEval
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

OneScience 社区提供可供训练的pdenneval数据集，用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确；

```bash
modelscope download --dataset OneScience/pdenneval  --local_dir ./data
```

### 训练

```bash
python scripts/train.py
```

### 训练权重
本仓库在weights/文件夹内提供基于pdenneval数据集训练的权重，即将上传该权重。

### 推理

```bash
python scripts/inference.py
```

推理默认读取 `weight/best_model.pt`，并将 `.npz` 结果写入 `result/output/`。

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

- PDENNEval 原始论文：[PDENNEval: A Comprehensive Evaluation of Neural Network Methods for Solving PDEs](https://www.ijcai.org/proceedings/2024/0573.pdf)。
- 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
