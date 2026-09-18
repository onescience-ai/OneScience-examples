<p align="center">
  <strong>
    <span style="font-size: 30px;">NABP-LSTM-Att</span>
  </strong>
</p>

# 模型介绍

NABP-LSTM-Att 是一个仅基于序列信息预测纳米抗体与抗原是否结合的二分类模型。模型分别编码纳米抗体 CDR 和抗原序列，通过一维卷积、双向 LSTM 与软注意力机制提取交互特征，最终输出 0～1 范围内的结合概率。

论文：[NABP-LSTM-Att: Nanobody–Antigen binding prediction using bidirectional LSTM and soft attention mechanism](https://doi.org/10.1016/j.compbiolchem.2025.108490)



# 模型描述

默认模型采用以下输入与结构：

- CDR 使用 3-mer 表示，输入长度为 24；
- 抗原使用 1-mer 表示，输入长度为 2371；
- CDR 序列嵌入与 CDR 编号嵌入相加后进入卷积层；
- 抗原序列经过独立嵌入与卷积层；
- 两路特征拼接后由 BiLSTM 和软注意力层聚合；
- Sigmoid 输出单个结合概率，使用二元交叉熵训练。



# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 纳米抗体–抗原结合预测 | 根据 CDR 和抗原序列特征输出结合概率 |
| 官方测试集评估 | 使用官方权重计算 AUROC 和 AUPR |
| 模型重新训练 | 使用预计算训练集和验证集从随机权重开始训练 |
| DCU 兼容性验证 | 检查推理、反向传播、权重更新及设备放置 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐在 OneScience 海光 DCU 环境中运行；
- 当前适配结果基于 BW DCU；
- 推理和训练均支持单卡运行。

### 下载模型包

~~~bash
python -m pip install modelscope
modelscope download --model OneScience/NABP-LSTM-Att --local_dir ./NABP-LSTM-Att
cd NABP-LSTM-Att
~~~

### 安装 OneScience 基础环境

~~~bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311

python -m pip install "onescience[bio-dcu]" \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
~~~

### 版本兼容性提示

本项目需升级DTK与Tensorflow。

加载 DTK 并安装增量依赖：

~~~bash
module load compiler/dtk/26.04
python -m pip install --upgrade --no-deps -r requirements.txt
~~~



### 权重与数据准备

模型包已包含推理和训练所需的预计算特征、k-mer 词表及官方权重，不需要另外下载数据。默认使用：

~~~text
conf/data/features/cdr_kmer3_ag_kmer1/
weight/cdr_kmer3_ag_kmer1/Model99.h5
~~~

### 快速推理

**作用：** 使用官方权重评估完整测试集，输出 AUROC、AUPR、耗时和吞吐量，并可保存预测结果。

~~~bash
python scripts/evaluate_test_set.py \
  --batch-size 64 \
  --output output/test_predictions.npz
~~~

### 完整数据集训练

**作用：** 从随机权重开始训练完整训练集，并使用验证集评估；默认训练 100 个 epoch，检查点写入 `output/checkpoints/`，不会覆盖官方权重。

~~~bash
python scripts/train.py
~~~

如只需验证一次完整训练链路，可执行一轮训练：

~~~bash
python scripts/train_one_epoch.py \
  --batch-size 64 \
  --output-dir output/one_epoch_run1
~~~

### 最小兼容性验证

~~~bash
python scripts/verify_minimal_inference.py --samples 8
python scripts/verify_minimal_training.py --batch-size 8 --steps 3
~~~

### 自定义数据

自定义序列需先转换成与官方 pickle 一致的 CDR 和抗原特征对象。数据获取、划分、k-mer TSV 生成及特征编码脚本位于 `scripts/`。从 SAbDab-nano 原始数据重新构建数据集时，还需准备 cd-hit 和 Clustal Omega。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：Ahmed, F. S., Aly, S., El-Tabakh, M. A. M., & Liu, X. (2025). [NABP-LSTM-Att: Nanobody–Antigen binding prediction using bidirectional LSTM and soft attention mechanism](https://doi.org/10.1016/j.compbiolchem.2025.108490). *Computational Biology and Chemistry*, **118**, 108490.
- 原始项目：[FMoonlightS/NABP-LSTM-Att](https://github.com/FMoonlightS/NABP-LSTM-Att)

- 本项目代码依据 MIT License 提供；使用模型、数据和外部工具时还应遵守各自的许可证与使用条款。
