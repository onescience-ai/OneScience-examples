<p align="center">
  <strong>
    <span style="font-size: 30px;">Geo-FNO</span>
  </strong>
</p>

# 模型介绍

Geo-FNO（Fourier Neural Operator with Learned Deformations）是面向一般几何体上的 PDE 求解的傅里叶神经算子，通过学习/给定的坐标变形把不规则物理域映射到均匀计算域并施加 FFT，从而求解参数化 PDE 的解算子 G: a->u。

论文：Fourier Neural Operator with Learned Deformations for PDEs on General Geometries (JMLR 2023)  
https://arxiv.org/abs/2207.05209

本仓库为论文复现版本，Tier1 交付弹性（hyper-elastic）问题：几何设计参数为 void 半径，输出为物理网格上的应力场。

# 模型描述

模型结构为 GeoFNO2d（论文公式 18）：

- `fc0`（Linear 提升）：`(B,N,2)` -> `(B,width,N)`
- 4 层 SpectralConv2d（modes=12, width=32）：
  - 第一层用几何傅里叶变换 Fa（公式 12，含 IPHI learned deformation）
  - 中间 2 层标准 rfft2
  - 最后一层用逆几何傅里叶变换 Fa^-1（公式 13）
  - 每层含 1x1 Conv 点态残差 W 与网格坐标生成的偏置 b
- `IPHI`（learned deformation）：3 层 FFN width=32，输入坐标 + code + sin/cos 高频特征，输出计算域坐标（公式 17）
- `fc1/fc2`（Q 投影）：`(B,width,N)` -> `(B,1,N)`

模型参数量约 1.19M。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| PDE 算子学习 | 学习参数化 PDE（弹性问题）的解算子 G: a -> u |
| 几何泛化 | 通过 learned deformation 处理任意物理域形状 |
| 模型训练 | 使用合成弹性数据训练 Geo-FNO |
| 模型推理 | 加载权重进行场预测 |
| 模型评估 | 计算相对 L2 error（论文公式 2） |

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
modelscope download --model OneScience/Geo-FNO --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem、bio、all（全领域）
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem（全领域）
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

Tier1 使用合成弹性数据（`model/data.py` 生成）：单位胞 [0,1]^2 内随机放置中心 void（半径 r=0.2+0.2/(1+exp(r̃))，约束 0.2<=r<=0.4），域内均匀采样约 1000 点并剔除 void 内点，合成平滑应力场作为 ground truth。

```bash
python scripts/train.py --config conf/default.yaml
```

### 训练

单卡：

```bash
python scripts/train.py --config conf/default.yaml
```

训练会在 `outputs/checkpoints/` 下保存 `model_best.pt` 与 `model.pt`。

### 训练权重

已上传的权重文件：

- `weight/model_best.pt`（best checkpoint，Tier1，test_rel_l2=0.9799）
- `weight/model.pt`（last epoch checkpoint）

### 推理

```bash
python scripts/evaluate.py --config conf/default.yaml \
    --checkpoint weight/model_best.pt
```

### 评估和可视化

```bash
python scripts/evaluate.py --config conf/default.yaml \
    --checkpoint weight/model_best.pt
```

评估输出逐样本相对 L2 error 报告（`outputs/evaluation_report.json` / `.md`）。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 Geo-FNO 原始论文的复现版本。

  Zongyi Li, Daniel Zhengyu Huang, Bilong Liu, Anima Anandkumar. "Fourier Neural Operator with Learned Deformations for PDEs on General Geometries." Journal of Machine Learning Research (JMLR), 2023. https://arxiv.org/abs/2207.05209

