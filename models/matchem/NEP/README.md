<p align="center">
  <strong>
    <span style="font-size: 30px;">NEP</span>
  </strong>
</p>

# 模型介绍

NEP（Neural Evolution Potential）是基于 MatPL 的神经网络势训练示例，面向原子结构数据学习能量、力和 virial 等材料相互作用信息。


# 模型描述

NEP 基于 MatPL 框架，使用 Cu、LiSiC 等原子体系数据进行训练，面向材料体系开展原子间势函数训练及分子动力学模拟。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| Cu 体系 NEP 训练 | 使用 `demo/nep_Cu/Cu_nep_train.json` 训练 Cu 势函数 |
| LiSiC 体系 NEP 训练 | 使用 `demo/nep_LiSiC/LiSiC_nep_train.json` 训练 LiSiC 势函数 |
| SLURM 作业提交 | 参考 `demo/nep_Cu/submit.sh` 在集群上运行训练 |
| 自有数据迁移 | 将客户数据整理为 `pwmat/movement` 等 MatPL 支持格式后替换训练路径 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 DCU 或 GPU 运行训练。
- CPU 可以用于导入和小配置连通性验证，完整训练速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/NEP --local_dir ./nep
cd nep
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

### 安装 MatPL

```bash
# 默认使用 test_pip 环境；若使用其他 conda 环境名，请先指定：
# export MATCHEM_CONDA_NAME=your_env
bash matpl_install.sh
```

### 训练数据介绍

本仓库不内置训练数据。以 MatPL 数据集为例，从 ModelScope 下载并放到仓库根目录的 `data/` 下：

```bash
modelscope download --dataset OneScience/MatPL --local_dir ./data
```

下载后数据路径为 `data/MatPL/`。

### 训练

单卡：

```bash
cd demo/nep_Cu
MatPL train Cu_nep_train.json
```

SLURM 提交：

```bash
cd demo/nep_Cu
bash submit.sh
```
### 训练权重

本仓库目前不内置训练权重，权重可通过上述训练得到。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- NEP 示例代码来自 OneScience 仓库。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。
- 如果在科研工作中使用 NEP 或 MatPL 训练结果，建议引用 OneScience 相关项目信息、MatPL/NEP 相关方法和实际使用的数据集来源。
