---
license: apache-2.0
language:
- en
- zh
tags:
- OneScience
- 地球科学
- 气象预报
frameworks: PyTorch
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">MultiTaskConvLSTM</span>
  </strong>
</p>


# 模型介绍

MultiTaskConvLSTM 是面向亚马逊流域降水临近预报的多任务 ConvLSTM 时空深度学习模型，以多层 ConvLSTM 作为基础骨干，融合风场、温湿度、土壤含水量、植被 LAI 等多源陆气网格变量，完成回归分支预测逐网格降水强度。


# 仓库说明

本仓库是OneScience整理的MultiTaskConvLSTM最小可运行模型仓库。

当前支持能力：

- 推理

当前不支持能力：

- 提供权重
- 内置真实数据下载
（需在https://huggingface.co/makkos-lilly/MultiTaskConvLSTM下载文件）
- 计算分类指标

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 模型推理 | 加载训练得到的权重，完成回归分支预测逐网格降水强度 |

# 主要文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 中文为主|
| `model/ConvLSTM.py` | 实现基础 ConvLSTM 单元（Cell）、多层 ConvLSTM 堆叠基础模块；提供时空卷积 LSTM 基础组件 |  |
| `model/MultiTaskConvLSTM.py` | 搭建多任务 ConvLSTM 完整网络：主干 ConvLSTM 编码器 + 双头输出 |  |
| `model/utils.py` | 封装各类气象时序预测评价指标 |  |
| `scripts/test_no_veg.py` | 无植被变量实验组推理评估入口脚本 | 9 通道输入，no_veg 专属数据集 + 权重 |
| `scripts/test_veg.py` | 植被变量实验组推理评估入口脚本 | 14 通道输入，veg 专属数据集 + 权重 |
| `weight/` | 权重目录 | 权重存放位置 |
| `config/` | 配置目录 | 该模型无配置文件 |
| `requirements.txt` | 依赖包  |  |

# 使用说明

容器镜像：flagos_earth_onecode:v1.0.0

## 快速开始

### 1. 下载模型包

```bash
# 默认下载到当前路径下MultiTaskConvLSTM文件夹，如需修改，则制定local_dir后的路径

modelscope download --model OneScience/MultiTaskConvLSTM --local_dir ./model
cd model
```

### 2. 使用方式
```bash
python scripts/test_no_veg.py
python scripts/test_veg.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

- 如果在科研工作中使用 MultiTaskConvLSTM 结果，建议引用 MultiTaskConvLSTM 原始论文和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。