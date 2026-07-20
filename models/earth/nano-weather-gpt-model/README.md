---
license: apache-2.0
language:
- en
- zh
tags:
- OneScience
- 地球科学
- 气象预报
- 短中期气象预报
frameworks: PyTorch
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">nano-Weather-GPT</span>
  </strong>
</p>


# 模型介绍

nano-Weather-GPT 是一个字符级轻量 GPT 模型，总参数量约 813K，基于合成瑞士城市天气预报文本从零训练。模型采用 4 层 Transformer 结构，包含 4 个注意力头、128 维嵌入向量，上下文长度 128 字符，输入任意英文气象开头语句，可自动续写长段天气预报文本，无需复杂气象网格数据，轻量化可在 CPU 环境直接推理。


# 仓库说明

本仓库是OneScience整理的nano-Weather-GPT最小可运行模型仓库。

当前支持能力：

- 推理
- 微调

当前不支持能力：

- 提供权重（需在https://huggingface.co/agoel7029/nano-weather-gpt-model下载权重文件）
- 内置真实数据下载与预处理


# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 模型推理 | 加载训练得到的权重，生成通顺、符合行业规范的天气预报文本 |
| 气象数据集微调适配 | 支持使用自有区域气象历史文本数据集做二次微调 |

# 主要文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 中文为主|
| `model/model.py` | 字符级GPT模型网络结构定义核心代码 | 推理、文本生成功能依赖该文件 |
| `model/data.py` | 数据集加载与文本预处理脚本 | 训练 / 微调流程使用，负责气象文本分词、字符 - ID 映射、训练集 / 验证集划分 |
| `model/canary.py` | 模型完整性校验脚本 | 部署 / 上传前校验权重、词表、模型代码是否完整，提前规避文件缺失、路径报错问题 |
| `scripts/test.py` | 模型推理测试Python脚本 | 输入气象提示词生成天气预报文本 |
| `scripts/train.py` | 微调脚本 | 加载数据集反向传播更新网络参数，需 GPU 加速；运行完成会生成新的权重文件 |
| `weight/` | 权重目录 | 权重存放位置 |
| `config/` | 配置目录 | 该模型无配置文件 |
| `requirements.txt` | 依赖包  |  |

# 使用说明

容器镜像：flagos_earth_onecode:v1.0.0

## 快速开始

### 1. 下载模型包

```bash
# 默认下载到当前路径下nano-weather-gpt-model文件夹，如需修改，则制定local_dir后的路径

modelscope download --model OneScience/nano-weather-gpt-model --local_dir ./model
cd model
```

### 2. 使用方式
- 推理
```bash
python scripts/test.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

- 如果在科研工作中使用 nano-Weather-GPT 结果，建议引用 nano-Weather-GPT 原始论文和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。