<p align="center">
  <strong>
    <span style="font-size: 30px;">DeePMD</span>
  </strong>
</p>

# 模型介绍

DeePMD 是面向原子体系机器学习势函数训练的 Deep Potential Molecular Dynamics 模型生态，基于 PyTorch/TensorFlow 后端提供最小可运行训练入口。

参考实现：DeepMD-kit 项目

# 模型描述

DeePMD 基于深度神经网络架构，使用原子体系数据进行训练，面向分子和材料体系开展原子间势函数训练及分子动力学模拟。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| DeePMD water 训练 | 使用 `demo/water_se_e2_a_pt/input_torch.json` 等配置训练 water 势函数 |
| 多卡 SLURM 提交 | 参考 `demo/water_se_e2_a_pt/submit_4card.sh`、`submit_8card.sh` |
| 环境连通性验证 | 运行 `dp_install.sh` 检查 DeepMD-kit 是否可安装 |
| 自有数据迁移 | 替换配置文件中的 `systems` 路径为自有 DeepMD npy 数据 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行训练。
- CPU 可以用于安装检查和小数据连通性验证，完整训练速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/DeePMD --local_dir ./deepmd
cd deepmd
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 安装 DeepMD-kit

```bash
# 默认使用 test_pip 环境；若使用其他 conda 环境名，请先指定：
# export MATCHEM_CONDA_NAME=your_env
bash dp_install.sh
```

### 训练数据介绍

本仓库不内置训练数据。以 DeePMD water 数据集为例，从 ModelScope 下载并放到仓库根目录的 `data/` 下：

```bash
modelscope download --dataset OneScience/DeePMD --local_dir ./data
```

下载后数据路径为 `data/DeePMD/water/data_0..3/`。

### 训练

单卡：

```bash
cd demo/water_se_e2_a_pt
dp --pt train input_torch.json
```

多卡：

```bash
cd demo/water_se_e2_a_pt
bash submit_4card.sh
```
### 训练权重

本仓库目前不内置训练权重，权重可通过上述训练得到。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- DeePMD 示例代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 DeepMD-kit 项目。上游 DeepMD-kit 相关许可请参考其官方仓库。
- 如果在科研工作中使用 DeePMD 训练结果，建议引用 DeepMD-kit、OneScience 相关项目信息和实际使用的数据集来源。
