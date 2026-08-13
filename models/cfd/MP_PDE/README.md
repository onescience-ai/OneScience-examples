<p align="center">
  <strong>
    <span style="font-size: 30px;">MP_PDE</span>
  </strong>
</p>

# 模型介绍
MP-PDE（Message Passing Neural PDE Solver）是由阿姆斯特丹大学等研究机构提出的一种基于消息传递机制的神经偏微分方程求解模型，发表于 ICLR 2022。该模型主要面向参数化偏微分方程（PDE）的时空演化预测任务，通过学习空间节点之间的局部相互作用实现 PDE 数值解的预测，并具备对不同方程参数、空间分辨率及网格结构的良好适应能力。

本项目基于 OneScience 技能，独立复现了 MP-PDE 论文中的 E3 参数化 PDE 预测实验，用于验证模型在不同 PDE 参数配置下对时空演化过程的预测能力。


论文：[Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376)


# 模型描述
MP-PDE 采用 MLP 编码器 + 多层消息传递图神经网络（GNN Processor）+ 一维 CNN 解码器 的 Encoder-Processor-Decoder 结构，将历史时间窗口内的 PDE 状态、空间坐标和方程参数编码为节点特征，通过多层消息传递学习局部空间关系，再结合 Temporal Bundling 一次预测后续多个时间步，实现自回归式 PDE 时空演化预测。



## 适用场景

| 场景 | 说明 |
| -- | -- |
| 跨分辨率 PDE 预测 | 基于图结构描述空间节点之间的关系，对固定卷积网格的依赖较小，可用于不同空间离散分辨率之间的训练与泛化 |
| 不规则网格计算 | 消息传递过程利用节点之间的相对空间位置，可以处理非均匀或不规则空间离散，在论文的 Wave Equation 实验中进行了相关验证 |
| 长时间自回归预测 | 通过 Temporal Bundling 和 Pushforward Trick 等方法减少自回归预测中的误差累积，提高长时间 PDE rollout 的稳定性 |




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
modelscope download --model OneScience/MP_PDE --local_dir ./MP_PDE
cd MP_PDE
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
本项目不依赖外部数据集，训练数据由 models/dataset.py 根据 config.yaml 在线合成。
默认输入函数 $u$ 从零均值高斯随机场（Gaussian Random Field，GRF）中采样，其协方差核定义为：

$$
k_l(x_1,x_2)
=
\exp\left(
-\frac{\lVert x_1-x_2\rVert^2}{2l^2}
\right),
$$

其中，相关长度设置为 $l=0.2$。输入函数首先在 1,000 个细网格点上生成，随后通过三次插值采样至 100 个等距传感器位置，并将这些离散采样值作为 MP_PDE 中 Branch Net 的输入。

每个监督学习样本表示为：

$$
\left(u,\ y,\ G(u)(y)\right),
$$

其中：

- $u$：输入函数在 100 个传感器位置上的离散取值；
- $y$：待预测的查询位置，其中 ODE 实验对应一维坐标，PDE 实验对应二维时空坐标 $(x,t)$；
- $G(u)(y)$：通过相应微分方程的数值求解器计算得到的参考解，表示算子 $G$ 作用于输入函数 $u$ 后在位置 $y$ 处的输出值。

### 训练

默认配置用于复现 MP-PDE 论文中的 E3 参数化偏微分方程实验。训练数据由 `models/dataset.py` 根据 `config/config.yaml` 在线生成，模型使用长度为 25 的历史时间窗口预测后续 25 个时间步。

运行 E3 训练：

```bash
python scripts/train.py \
    --config config/config.yaml \
    --generate-data \
    --device auto
```

### 训练权重

运行download.sh脚本，可下载复现训练得到的最优权重`weight/best_model.pth`，可直接用于推理微调；

### 推理

运行前需确保 E3 数据集 `data/e3.h5` 和训练权重 `weight/best_model.pth` 已存在。

```bash
python scripts/inference.py \
    --config config/config.yaml \
    --device auto
```

### 评估和可视化

完成训练和推理后，运行以下命令生成预测结果可视化：

```bash
python scripts/result.py \
    --config config/config.yaml
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 原论文链接：[Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376)
- 本项目为 MP_PDE 论文的独立复现，项目代码、模型权重、训练数据及第三方依赖分别适用其各自的许可条款
