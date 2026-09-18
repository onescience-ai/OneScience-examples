<p align="center">
  <strong>
    <span style="font-size: 30px;">CFDBench</span>
  </strong>
</p>

# 模型介绍

CFDBench 是由清华大学研究团队提出的面向计算流体力学机器学习方法的大规模评测基准，重点考察模型在不同边界条件、流体物性和几何构型下的泛化能力。

论文：[CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid Dynamics](https://arxiv.org/abs/2310.05963)

# 模型描述
CFDBench 基于多类典型计算流体力学数据构建，覆盖不同边界条件、流体物性和几何构型，面向机器学习方法开展流场预测性能与泛化能力评测。


# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| CFD代理模型基准评测 | 在统一数据划分和指标下比较不同神经算子或深度学习模型的流场预测能力 |
| 自回归流场演化预测 | 使用当前网格速度场逐步预测后续时刻的二维速度场，并观察多步误差累积 |
| 非自回归场查询 | 根据工况参数与时空坐标直接预测目标位置的速度，评估长时间范围查询能力 


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
```
modelscope download --model OneScience/CFDBench --local_dir ./CFDBench
cd CFDBench
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

OneScience 社区提供可供训练的 `cfdbench` 数据，用户可通过下述命令下载，并确认 `config/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/cfdbench --local_dir ./data
```

### 训练
**自回归训练**

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

**非自回归训练**
```bash
python scripts/train.py --model deeponet
```
### 训练权重
本仓库在weights/文件夹内提供 CFDBench 下的模型预训练权重，所有权重即将上传。

### 推理与可视化

推理脚本会按当前模型名自动选择任务类型，并默认读取：

```text
./weight/<model.name>.pt
```

默认 FNO 推理与可视化

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
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理；公开分发前请根据上游项目确认许可证要求。
