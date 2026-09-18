<p align="center">
  <strong>
    <span style="font-size: 30px;">Saluki</span>
  </strong>
</p>

# 模型介绍

Saluki 是一个用于预测哺乳动物 mRNA 半衰期的深度学习模型。模型读取完整 mRNA 序列，并联合编码密码子第一阅读框和剪接位点信息，通过卷积神经网络与门控循环单元学习影响 mRNA 稳定性的序列特征。



论文：[The genetic and biochemical determinants of mRNA degradation rates in mammals](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x)

# 模型描述

Saluki 的主要计算结构如下：

- 输入长度为 12,288，每个位置包含 6 个通道；
- 前 4 个通道表示 RNA 碱基序列；
- 第 5 个通道标记编码区的密码子第一阅读框；
- 第 6 个通道标记剪接位点；
- 使用多层一维卷积和最大池化提取局部序列特征；
- 使用 GRU 聚合长距离上下文信息；
- 使用全连接层输出 mRNA 半衰期预测值；
- data0/model0 和 data1/model1 使用共享结构及独立输出头；
- 训练采用 MSE、L2 正则和 Adam 优化器。

默认配置位于 conf/params.json。模型输入形状为 (batch, 12288, 6)，单个输出头的输出形状为 (batch, 1)。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| mRNA 半衰期预测 | 根据完整 mRNA 序列、编码框架和剪接信息预测相对稳定性 |
| 官方测试集评测 | 在 Saluki TFRecord 测试集上计算 MSE、Pearson r 和 R² |
| 双任务训练 | 同时使用 data0 和 data1 训练两个输出头 |
| 模型兼容性验证 | 检查 TensorFlow 是否能在 OneScience DTK/DCU 环境中建模和执行 forward |
| 序列特征研究 | 为后续梯度分析、ISM 和 motif 分析提供基础模型 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐在 OneScience DTK 环境中使用 DCU；
- 当前适配结果基于 BW DCU；
- 单卡推理和训练均可运行，建议至少预留 8 GB 设备内存；


### 下载模型包

安装 ModelScope 命令行工具后下载模型：

~~~bash
python -m pip install modelscope
modelscope download --model OneScience/Saluki --local_dir ./Saluki
cd Saluki
~~~

### 安装 OneScience 基础环境

~~~bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311

python -m pip install "onescience[bio-dcu]" \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
~~~

### 升级 DTK 与 TensorFlow

Saluki 的当前适配环境高于旧版 OneScience 默认 DTK/TensorFlow 组合。运行前需要先将平台 DTK 工具链升级到 **DTK 26.04**，再将 TensorFlow 升级到对应的 **DTK TensorFlow 2.18.0**。

DTK 属于平台编译器和运行时，请先按照 SCNet 平台说明切换到 DTK 26.04。随后执行一条命令安装对应的 DTK TensorFlow wheel 和 Saluki 额外依赖：

~~~bash
python -m pip install --no-deps -r requirements.txt
~~~

### 权重与数据准备

官方数据和预训练权重来自：

- 数据记录：[Zenodo 6326409](https://zenodo.org/records/6326409)
- 文件：datasets.zip
- 官方 MD5：45f0d6bd3857eb19e04eb5be2bb47451

相关数据权重结构目录如下：

~~~text
conf/data/f0_c0/
├── data0/
│   ├── statistics.json
│   └── tfrecords/
│       ├── train-*.tfr
│       ├── valid-*.tfr
│       └── test-*.tfr
└── data1/
    ├── statistics.json
    └── tfrecords/
        ├── train-*.tfr
        ├── valid-*.tfr
        └── test-*.tfr

weight/f0_c0/
├── model0_best.h5
└── model1_best.h5
~~~




### 快速推理

**model0 / data0**

**作用：** 使用第 0 个输出头和官方 model0 权重，为 data0 测试集中的每条 mRNA 生成一个半衰期预测分数。该分数表示模型根据碱基序列、编码框架和剪接位点判断的相对 mRNA 稳定性，通常分数越高表示预测半衰期越长；它是按官方 data0 目标预处理方式学习的回归值，不能直接解释为小时。输出同时包含真实目标与预测值，以及 MSE、Pearson r 和 R²；MSE 越低、Pearson r 和 R² 越高，表示预测与实验目标越一致。

~~~bash
python scripts/predict.py \
  conf/data/f0_c0/data0 \
  weight/f0_c0/model0_best.h5 \
  --head 0 \
  --out-dir output/f0_c0/model0
~~~

**model1 / data1**

**作用：** 使用第 1 个输出头和官方 model1 权重，为 data1 测试集中的每条 mRNA 生成一个半衰期预测分数。该分数表示模型在 data1 目标体系下预测的相对 mRNA 稳定性，通常分数越高表示预测半衰期越长；data0 与 data1 分别使用独立输出头，因此两组原始分数不应脱离各自数据集直接横向比较。输出同时保存真实目标、预测值、MSE、Pearson r 和 R²，用于衡量预测误差、排序一致性和对目标方差的解释程度。

~~~bash
python scripts/predict.py \
  conf/data/f0_c0/data1 \
  weight/f0_c0/model1_best.h5 \
  --head 1 \
  --out-dir output/f0_c0/model1
~~~

每个输出目录包含：

~~~text
predictions.h5
metrics.json
~~~

predictions.h5 保存预测值和目标值，metrics.json 保存样本数、输出形状、dtype、NaN/Inf 检查、MSE、Pearson r 和 R²。

### 完整数据集训练

默认配置包含两个输出头，因此训练时应同时传入 data0 和 data1：

**作用：** 同时读取 data0 和 data1 的训练及验证 TFRecord，优化共享的一维卷积—GRU 特征提取主干以及两个独立回归输出头，使模型学习从 6 通道 mRNA 表征到两个数据集半衰期目标分数的映射。训练过程使用验证集损失选择并保存 model0 和 model1 的最佳权重与恢复检查点；这些文件可用于后续推理、继续训练和比较不同训练轮次，但单个 epoch 仅用于验证训练链路，不代表达到论文最终精度。

~~~bash
python scripts/train.py \
  conf/data/f0_c0/data0 \
  conf/data/f0_c0/data1 \
  --params conf/params.json \
  --out-dir output/f0_c0/train
~~~

训练入口执行真实 TFRecord 数据读取、forward、MSE/L2 loss、backward、Adam 更新、验证及 checkpoint 保存。

如只验证 1 epoch，请复制 conf/params.json 为新的相对配置文件，并将 train_epochs_min、train_epochs_max 和 patience 分别设为 1、1 和 0。不要直接覆盖默认参数文件，以便保留正式训练配置。

### 自定义数据

自定义数据必须转换成 Saluki 使用的压缩 TFRecord 格式，并为每个样本提供：

- RNA 碱基序列；
- 实际序列长度；
- 密码子第一阅读框轨道；
- 剪接位点轨道；
- 回归目标。

每个数据集目录必须包含 statistics.json 和 tfrecords/。可复制 conf/data/f0_c0 的相对目录形式组织新数据。



# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Saluki 论文：[Agarwal and Kelley, Genome Biology, 2022](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x)
- Saluki 论文复现代码：[vagarwal87/saluki_paper](https://github.com/vagarwal87/saluki_paper)
- Basenji 官方实现：[calico/basenji](https://github.com/calico/basenji)
- 官方数据：[Zenodo 6326409](https://zenodo.org/records/6326409)
- 本模型包中的 Basenji/Saluki 代码依据 Apache-2.0 许可证提供，使用模型、代码和数据时还应遵守其各自许可证与使用条款。
