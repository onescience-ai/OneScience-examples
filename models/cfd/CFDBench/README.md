
<p align="center">
  <strong>
    <span style="font-size: 30px;">CFDBench</span>
  </strong>
</p>

# 模型介绍

CFDBench 是面向计算流体力学机器学习方法评测的大规模基准，用于考察模型在不同边界条件、流体物性和几何形状下的泛化能力。本标准运行包适配原始仓库的两类入口：自回归模型根据当前二维速度场预测下一时刻速度场，非自回归模型根据工况参数与时空坐标查询场值。

# 仓库说明

本仓库是 OneScience 整理的 CFDBench 最小可运行标准工程，面向本地训练、推理和快速验证场景。

当前支持能力：

- 生成 tiny fake data 进行流程验证
- 自回归模型训练与测试
- 非自回归模型训练与测试
- checkpoint 推理
- 评估与可视化

当前不支持能力：

- 不内置预训练权重
- 不负责自动下载、清洗或重新适配全部外部数据库
- 不包含原始 CFD 求解器、网格生成或数据清洗流程

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| CFD替代模型基准 | 在统一数据划分和指标下比较不同神经算子或深度学习模型的流场预测能力 |
| 未见工况泛化评估 | 检验模型对训练阶段未出现的边界条件、流体密度与黏度、计算域几何的泛化能力 |
| 典型流动问题研究 | 研究顶盖驱动方腔流、圆管流、坝流和圆柱绕流中的边界层、射流及涡脱落等现象 |
| 自回归流场推进 | 使用当前网格速度场逐步预测后续时刻的二维速度场，并观察多步误差累积 |
| 非自回归场查询 | 根据工况参数与时空坐标直接预测目标位置的速度，评估长时间范围查询能力 

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | 工程元信息 | 最小配置 |
| `conf/config.yaml` | 数据、模型、训练和推理配置 | 默认 tiny smoke test 配置 |
| `scripts/fake_data.py` | 假数据生成脚本 | 生成 `tube/prop`、`tube/bc`、`tube/geo` case 目录 |
| `scripts/train_auto.py` | 自回归训练脚本 | 支持 `auto_*`、`resnet`、`unet`、`fno` |
| `scripts/train.py` | 非自回归训练脚本 | 支持 `ffn`、`deeponet` |
| `scripts/inference.py` | 推理脚本 | 读取 `root.inference.checkpoint_path` 或自动权重 |
| `scripts/result.py` | 结果查看脚本 | 打印预测张量形状、范围和指标 |
| `model/` | 模型文件包 | OneScience复现的经典TOP模型|
| `weight/` | 权重目录 | 默认按模型名保存 `<model.name>.pt` |

## 支持模型

| 类型 | `root.model.name` | 训练入口 |
| :--- | :--- | :--- |
| 非自回归 | `ffn` | `python scripts/train.py` |
| 非自回归 | `deeponet` | `python scripts/train.py` |
| 自回归 | `auto_ffn` | `python scripts/train_auto.py` |
| 自回归 | `auto_deeponet` | `python scripts/train_auto.py` |
| 自回归 | `auto_edeeponet` | `python scripts/train_auto.py` |
| 自回归 | `auto_deeponet_cnn` | `python scripts/train_auto.py` |
| 自回归 | `resnet` | `python scripts/train_auto.py` |
| 自回归 | `unet` | `python scripts/train_auto.py` |
| 自回归 | `fno` | `python scripts/train_auto.py` |

脚本会根据 `root.model.name` 自动设置 OneScience `CFDBenchDatapipe` 的 `task_type`。`auto_deeponet_cnn` 要求四次池化后特征图为 `4x4`；使用默认 tube fake data 时，请将 `root.datapipe.data.num_rows` 和 `root.datapipe.data.num_cols` 设为 `64`。
# 使用说明

## 1. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

## 2. 快速开始

### 安装运行环境

```bash
git clone https://gitee.com/onescience-ai/onescience.git
cd onescience
bash install.sh cfd
```

### 生成假数据进行流程验证

如需先用最小假数据检查路径和数据格式，保持 `config/config.yaml` 的默认配置即可

```bash
python scripts/fake_data.py
```
如需使用真实数据，请准备 CFDBench 数据集，并将 `conf/config.yaml` 中的数据路径指向数据集的 `data` 目录：

```yaml
root:
  datapipe:
    source:
      data_dir: "/path/to/OneScience_cfdbench/data"
```


### 自回归训练

默认配置为 `root.model.name: fno`，因此使用自回归入口：

```bash
python scripts/train_auto.py
```


切换其他自回归模型可修改 `conf/config.yaml`：

```yaml
root:
  model:
    name: "auto_ffn"
```

## 非自回归训练

非自回归入口为 `scripts/train.py`，通过 `--model` 参数选择模型：

```bash
python scripts/train.py --model ffn
python scripts/train.py --model deeponet
```


## 推理与结果查看

推理脚本会按当前模型名自动选择任务类型，并默认读取：

```text
./weight/<model.name>.pt
```

默认 FNO 推理：

```bash
python scripts/inference.py
python scripts/result.py
```

非自回归模型推理同样通过参数选择：

```bash
python scripts/inference.py --model ffn
python scripts/result.py --model ffn
```
# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- CFDBench 原始论文：[CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid Dynamics](https://arxiv.org/abs/2310.05963)
- CFDBench 原始代码仓库：https://github.com/luo-yining/CFDBench
- 本仓库保留来源说明，并面向 OneScience 本地运行场景进行整理；公开分发前请根据上游项目确认许可证要求。
