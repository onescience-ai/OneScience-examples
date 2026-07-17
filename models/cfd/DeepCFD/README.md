<p align="center">
  <strong>
    <span style="font-size: 30px;">DeepCFD</span>
  </strong>
</p>

# 模型介绍
DeepCFD 是德国航空航天中心 DLR研究人员提出的用于二维非均匀稳态层流近似求解的深度卷积神经网络模型。该模型以流场几何信息为输入，包括障碍物的有符号距离函数 SDF 和流动区域类别标记，通过 U-Net 编码器提取潜在几何表示，再由解码器重建完整的速度场和压力场，用于替代传统 CFD 求解器快速预测稳态层流结果。它主要适用于二维通道流、低雷诺数层流、绕障碍物流动预测、工程早期设计筛选、气动外形快速评估和传统 CFD 代理求解加速等场景。


# 仓库说明

本仓库是 OneScience 整理的 DeepCFD 标准运行包，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：
* 训练
* 推理
* 评估与可视化
* 生成空壳假数据用于流程连通性验证


当前不支持能力：
* 不内置预训练权重
* 不负责自动下载、清洗或重新适配全部外部数据库
* 不内置 OpenFOAM 等传统 CFD 求解器，也不负责重新生成真实 CFD 标签


## 适用场景

| 场景         | 说明                                 |
| ---------- | ---------------------------------- |
| 稳态层流预测     | 近似求解二维非均匀稳态层流中的速度场和压力场             |
| 通道绕流仿真     | 预测通道内随机形状障碍物周围的流场分布                |
| CFD 代理求解   | 替代 OpenFOAM 等传统 CFD 求解流程，提高推理效率    |
| 气动/流体外形优化  | 用于需要大量几何候选方案评估的低速层流设计任务            |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | OneCode 元信息 | 保持最小配置 |
| `config/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train.py` | 训练脚本 | 支持单卡和 torchrun 多卡 |
| `scripts/inference.py` | 推理脚本 | 需存在训练权重 |
| `scripts/result.py` | 结果查看脚本 | 读取 checkpoint 和推理输出数组并打印摘要 |
| `scripts/fake_data.py` | 假数据生成脚本 | 用于快速连通性验证 |
| `model/` | 模型文件包 | OneScience复现的经典TOP模型 |
| `weight/` | 权重目录 | 训练默认保存 `best_model.pt` |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


## 3. 快速开始

### 安装运行环境

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 生成假数据进行流程验证

如需先用最小假数据检查路径和数据格式，保持 `config/config.yaml` 的默认配置即可。脚本会根据 `root.fake_data` 配置生成 `dataX.pkl` 和 `dataY.pkl`，默认写入 `./data/deepcfd/`。

```bash
python scripts/fake_data.py
```

此项目的数据集可以使用以此链接[下载](https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1)
数据集包含两个文件：
- `dataX.pkl`: 981个管道流样本的几何输入信息
- `dataY.pkl`: 对应样本的真实CFD解（使用simpleFOAM求解器计算）

### 训练

```bash
python scripts/train.py
```

训练参数来自 `config/config.yaml` 的 `root.datapipe`、`root.model` 和 `root.training` 字段。默认配置用于最小 smoke test；真实训练时请将 `root.datapipe.source.data_dir` 指向真实 DeepCFD 数据目录，并按需要调大模型规模、batch size 和 epoch 数。

默认训练会保存最佳 checkpoint：

```text
./weight/best_model.pt
```

多卡训练可使用 `torchrun` 或集群调度器设置的分布式环境变量启动，例如：

```bash
torchrun --standalone --nproc_per_node=<num_GPUs> scripts/train.py
```

### 推理

```bash
python scripts/inference.py
```

推理会读取 `root.inference.checkpoint_path` 指向的训练权重，默认是：

```text
./weight/best_model.pt
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证
- DeepCFD原始论文：[DeepCFD: Efficient Steady-State Laminar Flow
Approximation with Deep Convolutional Neural](https://arxiv.org/abs/2004.08826)。
- 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
