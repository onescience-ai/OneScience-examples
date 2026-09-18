<p align="center">
  <strong>
    <span style="font-size: 30px;">BENO</span>
  </strong>
</p>

# 模型介绍

BENO 是北京大学相关团队提出的用于复杂边界条件下椭圆型 PDE 求解的边界嵌入神经算子，可对不同几何区域和非齐次边界条件下的稳态物理场分布进行快速预测。

论文：[BENO: Boundary-embedded Neural Operators for Elliptic PDEs](https://openreview.net/forum?id=ZZTkLDRmkg)

# 模型描述
BENO 基于融合双分支图神经网络与 Transformer 的边界嵌入神经算子架构，使用beno数据集训练，面向复杂几何及非齐次边界条件开展 Poisson、Laplace 等椭圆型偏微分方程的稳态物理场预测。

# 适用场景

| 场景         | 说明                                |
| ---------- | --------------------------------- |
| 椭圆型 PDE 求解 | 面向 Poisson、Laplace 等稳态边值问题的快速近似求解 |
| 复杂边界条件建模   | 处理自由形状边界、不规则区域和非齐次边界值对解场的影响       |
| 稳态物理场预测    | 预测由源项和边界共同决定的平衡态物理场分布             |
| 数值求解器代理加速  | 替代或辅助 FEM、FDM、FVM 等传统求解流程，提高推理效率  |

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
modelscope download --model OneScience/BENO --local_dir ./BENO
cd BENO
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
OneScience 社区提供可供训练的 `beno` 数据，用户可通过下述命令下载，并确认 `config/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/beno --local_dir ./data
```
完整的数据集文件也可通过[官方链接](https://drive.google.com/file/d/11PbUrzJ-b18VhFGY_uICSciCkeGrsaTZ/view)下载

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：
```
torchrun --standalone --nnodes=<num_nodes> --nproc_per_node=<num_GPUs> scripts/train.py
```

### 训练权重
本仓库在weights/文件夹内提供基于 beno 数据训练的权重，该权重即将上传。

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

- BENO 原始论文：[BENO: Boundary-embedded Neural Operators for Elliptic PDEs](https://proceedings.iclr.cc/paper_files/paper/2024/file/218ca0d92e6ed8f9db00621e103dc70c-Paper-Conference.pdf)
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理；公开分发前请根据上游项目确认许可证要求。
