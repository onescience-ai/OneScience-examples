<p align="center">
  <strong>
    <span style="font-size: 30px;">Earthformer</span>
  </strong>
</p>

# 模型介绍

Earthformer 由亚马逊云服务（AWS）的研究人员联合香港科技大学提出，旨在解决传统Transformer在处理高维地球物理数据时计算成本过高的问题。

Earthformer: Exploring Space-Time Transformers for Earth System Forecasting

https://arxiv.org/abs/2207.05833



# 模型描述

Earthformer 是一个为地球系统（如天气和气候）预报设计的时空Transformer模型，其核心结构是一个名为“Cuboid Attention”的新型注意力机制。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 SEVIR 结构数据训练 Earthformer 气象预报模型。 |
| 本地快速验证 | 使用虚拟数据检查数据读取，模型训练、推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |



# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)


## 2. 手动安装使用

以下命令均假定在 Earthformer 项目根目录执行。

**硬件要求**

- 训练和推理必须使用 PyTorch 可识别的 GPU 或 DCU；CPU 可用于生成虚拟数据和检查配置，但不能运行当前训练与推理脚本。
- 多卡训练使用 NCCL 后端，请确保设备驱动、通信库和 PyTorch 版本匹配。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/Earthformer --local_dir ./Earthformer
cd Earthformer
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```



### 生成虚拟数据

虚拟数据只用于检查数据协议和程序流程，不代表真实 SEVIR 数据或模型预报效果：

```bash
python script/fake_data.py
```

默认生成 `data/synthetic_sevir/{train,val,test}.npz` 和 `metadata.json`。如需生成与官方 SEVIR 空间尺寸一致的合成数据：

```bash
python script/fake_data.py --output-dir data/synthetic_sevir_384 --height 384 --width 384
```

### 训练

单卡：

```bash
python script/train.py
```


多卡：

```bash
torchrun --nproc_per_node=8 script/train.py
```

训练从随机初始化开始，默认保存到 `data/checkpoint/earthformer.pt`。

### 训练权重

本仓库在 weights/ 文件夹内提供基于 SEVIR 数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

推理默认读取训练检查点，输出文件为 `output/predictions.npz`。

```bash
python script/inference.py
```


### 评估和可视化

```bash
python script/result.py
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证


- 本仓库为 Earthformer 原始论文的 OneScience 复现版本。Earthformer 官方实现采用 Apache License 2.0；本仓库代码和 SEVIR 数据的使用仍应以各自项目中的许可证及使用条款为准。

