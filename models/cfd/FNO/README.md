<p align="center">
  <strong>
    <span style="font-size: 30px;">FNO</span>
  </strong>
</p>

# 模型介绍
FNO（Fourier Neural Operator，傅里叶神经算子）是一类面向参数化偏微分方程的神经算子，通过在 Fourier 空间参数化积分核，直接学习输入函数到解函数的映射。本项目依据Onescience 技能独立复现 FNO-2D 的二维不可压 Navier–Stokes 涡量预测实验。

论文：[Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

# 模型描述
本实现以连续 10 帧、分辨率为 64×64 的涡量场为输入，并递归预测后续 10 帧。模型先将历史场与二维周期坐标映射到宽度为 32 的隐空间，再依次通过 4 个 Fourier Layer；每层保留两个空间方向各 12 个 Fourier 模态，将谱卷积与 1×1 局部卷积相加后执行 BatchNorm 和 ReLU。输出头采用 32→128→1 的投影，逐步生成下一时刻涡量场，并将预测结果回填至输入窗口完成闭环推理。


## 适用场景

| 场景 | 说明 |
| --- | --- |
| 参数化 PDE 算子学习 | 学习PDE参数、系数场或初始条件到对应解场的映射，适用于需要针对大量不同参数重复求解PDE的场景。 |
| Burgers方程预测 | 根据一维Burgers方程的初始条件预测未来状态，用于验证模型对非线性演化方程的算子学习能力。 |
|DarcyF1ow 预测 | 根据二维介质的扩散或渗透系数场预测对应稳态解，可用于多孔介质流动、地下渗流等问题。|
| Navier–Stokes 流场预测 | 根据历史涡量场递归预测二维不可压缩流体的后续演化。 |


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
modelscope download --model OneScience/FNO --local_dir ./FNO
cd FNO
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
实验使用 `NavierStokes_V1e-5_N1200_T20.mat`。其中shape为`[N,H,W,T]=[1200,64,64,20]`；每条轨迹以前 10 帧作为输入、后 10 帧作为预测目标。前 1000 条轨迹用于训练，后 200 条用于测试，不设置独立验证集，也不执行归一化。

可通过下述命令下载数据：

```bash
modelscope download --dataset OneScience/fno --local_dir ./data
```

下载后，将 `config/config.yaml` 中的 `data.root` 指向数据目录，并确认 `data.file` 为上述 MAT 文件名。训练脚本会严格校验字段名、shape、dtype 和有限值。

### 训练

默认配置对应论文中 `ν=1e-5`、`T=20` 的 FNO-2D 实验：训练 500 个 epoch，batch size 为 20，使用 Adam（初始学习率 `1e-3`），每 100 个 epoch 将学习率减半。

```bash
python scripts/train.py --config config/config.yaml --device auto
```

训练集全轨迹相对 L2 最优的权重保存为 `weight/best_model.pth`，每个 epoch 的最新完整训练状态保存为 `weight/last_model.pth`；训练历史写入 `results/train_history.json`。



### 训练权重

运行download.sh脚本，可下载复现训练得到的最优权重`weight/best_model.pth`，可直接用于推理微调；

### 推理

运行前需确保配置中的数据路径有效，且 `weight/best_model.pth` 已存在。正式推理固定使用配置中的 200 条测试轨迹和 10 步闭环预测，并实时打印批次进度与最终相对 L2：

```bash
python scripts/inference.py --config config/config.yaml 
```

推理结果保存为：

- `results/predictions.npz`：预测值、真实值、样本索引和预测时刻；
- `results/metrics.json`：总体、逐预测步指标及论文参考值对比；
- `results/per_sample_metrics.csv`：逐样本相对 L2。

### 评估和可视化

完成训练和推理后运行：

```bash
python scripts/result.py --config config/config.yaml --sample-index 0
```

脚本会从 `predictions.npz` 重新计算指标，并交叉校验 JSON、CSV、最佳 epoch 和预测 shape，随后生成：

- `results/training_curves.png`：训练/测试误差及训练目标曲线；
- `results/sample_000_rollout.png`：代表样本在 `t=11、16、20` 的真实场、预测场和绝对误差；
- `results/run_metadata.json`：配置、运行环境、假设、文件哈希和质量检查；
- `results/summary.md`：实验结果摘要。


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文链接：[Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- 本模型包代码采用 MIT License。使用模型权重时还应遵守训练数据及第三方依赖的许可与适用条款。
