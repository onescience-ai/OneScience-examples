
<p align="center">
  <strong>
    <span style="font-size: 30px;">DeepONet</span>
  </strong>
</p>

# 模型介绍
DeepONet 是由布朗大学（Brown University）相关研究团队提出的一种面向算子学习（Operator Learning）的深度神经网络模型，并于 2021 年发表于 Nature Machine Intelligence。与传统神经网络主要学习有限维向量之间的映射关系不同，DeepONet 旨在直接学习函数空间到函数空间之间的非线性算子映射，从而建立输入函数与相应输出函数之间的关系。该模型可用于逼近由常微分方程、偏微分方程以及其他物理系统所定义的解算子，为复杂动力系统和科学计算问题提供高效的数据驱动建模方法。

本项目基于 OneScience 技能，独立复现了 DeepONet 论文相关实验。


论文：[DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators](https://arxiv.org/abs/1910.03193)


# 模型描述
DeepONet 采用由 Branch Net（分支网络）和 Trunk Net（主干网络）组成的双网络结构，Branch Net 编码输入函数在固定传感器位置上的离散采样值，Trunk Net 编码待预测位置的空间或时空坐标，最终通过两组特征向量的内积并加入偏置得到目标算子在指定位置的输出。




## 适用场景

| 场景        | 说明                                                |
| --------- | ------------------------------------------------- |
| 算子学习 | 用于学习函数空间之间的映射关系，即由输入函数 (u) 直接预测输出函数 (G(u))，适用于传统神经网络难以直接处理的函数到函数映射问题                      |
| 时空场预测   | 可用于 Navier–Stokes、Compressible Euler 等流体问题的快速代理预测 |
| 多尺度物理场建模  | Trunk Net 可以直接输入 ((x,t)) 等多维坐标，因此适用于温度场、浓度场、扩散场等随空间和时间变化的物理场预测                         |
| 多查询点预测    | 对同一个输入函数，只需固定 Branch Net 的输入，通过改变 Trunk Net 的查询坐标即可预测不同空间或时间位置上的解                   |




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
modelscope download --model OneScience/DeepONet --local_dir ./DeepONet
cd DeepONet
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

本项目不依赖外部数据集，E3 数据由 `models/dataset.py` 根据 `config/config.yaml` 在线生成，对应参数化一维 PDE：

$$
\frac{\partial u}{\partial t}
+\alpha\frac{\partial(u^2)}{\partial x}
-\beta\frac{\partial^2u}{\partial x^2}
+\gamma\frac{\partial^3u}{\partial x^3}
=\delta(t,x),
$$

其中 $x\in[0,16)$、$t\in[0,4]$，空间方向采用周期边界条件。方程参数独立采样：

$$
\alpha\sim\mathcal U(0,3),\quad
\beta\sim\mathcal U(0,0.4),\quad
\gamma\sim\mathcal U(0,1).
$$

外力及初始条件定义为：

$$
\delta(t,x)=\sum_{j=1}^{5}
A_j\sin\left(
\omega_jt+\frac{2\pi k_jx}{16}+\phi_j
\right),
\qquad
u(0,x)=\delta(0,x),
$$

其中：

- $A_j\sim\mathcal U(-0.5,0.5)$；
- $\omega_j=-0.4$；
- $k_j\in\{1,2,3\}$；
- $\phi_j\sim\mathcal U(0,2\pi)$。

参考解首先在 200 个空间网格点上生成，随后下采样至 100 个网格点。非线性通量采用五阶 WENO 格式，时间推进采用四阶 Runge–Kutta 方法。每条轨迹包含 250 个时间点，数据形状为：

$$
u\in\mathbb R^{250\times100}.
$$

模型使用长度为 $K=25$ 的历史窗口预测后续 25 个时间步。单个监督样本可表示为：

$$
\left(
u_{i:i+K-1},\,x,\,t,\,(\alpha,\beta,\gamma);
\ u_{i+K:i+2K-1}
\right).
$$


### 训练

默认配置复现 DeepONet 论文中的四组算子学习实验，包括反导数、非线性 ODE、受迫摆和扩散–反应方程。训练脚本默认运行反导数实验，输入函数和参考解由 `models/dataset.py` 在线生成。

运行默认反导数实验：

```bash
python scripts/train.py \
    --config config/config.yaml \
    --experiment antiderivative \
    --device auto
```
将 --experiment 设置为 all，可依次运行论文中的全部四项主实验

训练过程中按照 config/config.yaml 中的间隔实时输出训练 loss、测试 MSE 和相对 L2 误差。测试 MSE 改善时，模型权重、优化器状态、当前迭代次数、评测指标及实际运行配置会保存至 weight/best_model.pth。

### 训练权重

运行download.sh脚本，可下载复现训练得到的最优权重`weight/best_model.pth`，可直接用于推理微调；

### 推理

运行前需确保配置中的数据路径有效，且 `weight/best_model.pth` 已存在。

```bash
python scripts/inference.py \
    --config config/config.yaml \
    --experiment <实验名称> \
    --variant unstacked_bias \
    --mode <推理模式> \
    --device auto
```
推理模式包括：
- random_test：随机测试集；
- ood：ODE 分布外输入；
- pde_grid：扩散–反应二维时空场。
默认 batch size 为 8192，可通过 --batch-size 修改。预测结果和评测指标保存至 results 目录。

### 评估和可视化

完成训练和推理后，可汇总已有实验结果并生成训练曲线、预测对比图和评测报告：

```bash
python scripts/result.py --config config/config.yaml 
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 原论文链接：[DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators](https://arxiv.org/abs/1910.03193)
- 本项目为 DeepONet 论文的独立复现。官方实现代码采用 MIT License；本项目代码、模型权重、训练数据及第三方依赖分别适用其各自的许可条款
