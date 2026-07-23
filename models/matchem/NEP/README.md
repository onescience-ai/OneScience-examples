# NEP

NEP（Neural Evolution Potential）是基于 MatPL 的神经网络势训练示例，面向原子结构数据学习能量、力和 virial 等材料相互作用信息。本目录是 OneScience MatChem 领域整理的 NEP 最小可运行训练示例，提供 Cu、LiSiC 等体系的训练配置和提交脚本。

---

## 目录说明

本目录是 OneScience 主仓库 `examples/matchem/nep` 的示例，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 训练：使用 MatPL 训练 NEP 原子间势。
- 分布式训练：支持单卡 SLURM 提交脚本。
- 多体系示例：保留 Cu、LiSiC 等上游示例配置。

当前不支持能力：

- 不内置预训练权重或 checkpoint。
- 不内置训练数据，需从同 ID 数据集仓库 `OneScience/MatPL` 下载。
- 不提供独立推理服务、部署脚本或可视化页面。

---

## 适用场景

| 场景 | 说明 |
| :---: | :---: |
| Cu 体系 NEP 训练 | 使用 `demo/nep_Cu/Cu_nep_train.json` 训练 Cu 势函数 |
| LiSiC 体系 NEP 训练 | 使用 `demo/nep_LiSiC/LiSiC_nep_train.json` 训练 LiSiC 势函数 |
| SLURM 作业提交 | 参考 `demo/nep_Cu/submit.sh` 在集群上运行训练 |
| 自有数据迁移 | 将客户数据整理为 `pwmat/movement` 等 MatPL 支持格式后替换训练路径 |

---

## 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 本文件 |
| `matpl_install.sh` | MatPL 安装脚本 | 在 OneScience matchem 环境基础上安装 MatPL |
| `demo/nep_Cu/` | Cu 单卡训练示例 | 含 `Cu_nep_train.json`、`std_input.json`、`submit.sh` |
| `demo/nep_LiSiC/` | LiSiC 单卡训练示例 | 含 `LiSiC_nep_train.json`、`std_input.json`、`submit.sh` |

---

## 使用说明

### 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

### 2. 手动安装使用

**硬件要求**

- 推荐使用 DCU 或 GPU 运行训练。
- CPU 可以用于配置检查，不建议用于正式 NEP 训练。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**软件要求**

- Python 3.11
- numpy
- MatPL
- OneScience matchem 运行环境

安装运行环境：

DCU环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

GPU环境

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

安装 MatPL：

```bash
# 默认使用 test_pip 环境；若使用其他 conda 环境名，请先指定：
# export MATCHEM_CONDA_NAME=your_env
source matchem_env.sh
bash matpl_install.sh
```

> 安装脚本会自动检查并通过 conda-forge 安装 `gflags`/`glog`（torch cmake 的运行时依赖）；从 github 下载 glog 源码时优先 curl，失败自动回退 wget。

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光 DCU：

```bash
hy-smi
```

### 3. 快速开始

**进入示例目录**

本示例位于 OneScience 仓库的 `examples/matchem/nep`，进入该目录后所有命令均相对于该目录执行：

```bash
cd examples/matchem/nep
```

**准备数据**

本目录不内置训练数据。以 MatPL 数据集为例，从 ModelScope 下载并放到目录的 `data/` 下：

```bash
modelscope download --dataset OneScience/MatPL --local_dir ./data
```

下载后数据路径为 `data/MatPL/`。

**运行样例训练**

以 Cu 体系为例：

```bash
cd demo/nep_Cu
MatPL train Cu_nep_train.json
```

SLURM 提交：

```bash
cd demo/nep_Cu
bash submit.sh
```

---

## OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

---

## 引用与许可证

- NEP 示例代码来自 OneScience 仓库。本仓库保留来源说明，并面向 OneScience 社区使用场景进行整理。
- 如果在科研工作中使用 NEP 或 MatPL 训练结果，建议引用 OneScience 相关项目信息、MatPL/NEP 相关方法和实际使用的数据集来源。
