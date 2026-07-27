---
license: apache-2.0
language:
- en
- zh
tags:
- OneScience
- 地球科学
- 短时预报
frameworks: PyTorch
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">AI-GAMFS</span>
  </strong>
</p>


# 模型介绍

AI-GAMFS是基于深度学习GEOS-FP气溶胶快速预报系统，是用来替代传统数值模式（GEOS Chem/GEOS-FP）做气溶胶场短时预报的AI气象/气溶胶模型。


# 仓库说明

本仓库是OneScience整理的AI-GAMFS最小可运行模型仓库。

当前支持能力：

- 推理

当前不支持能力：

- 提供权重、utils文件夹、info.pkl等等
（需在https://huggingface.co/zhangxutao/AI-GAMFS和https://github.com/zhangxutao3/AI-GAMFS.git下载文件）

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 模型推理 | 气溶胶-气象耦合人工智能预报 |

# 主要文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 中文为主|
| `model/scheduler.py` | 滚动预测逻辑脚本 | 提供 rolling_model 滚动预测函数，执行多步接力预报 |
| `scripts/test.py` | 推理脚本 | |
| `weight/` | 权重目录 | 权重存放位置 |
| `config/` | 配置目录 | 该模型无配置文件 |
| `requirements.txt` | 依赖包  |  |

# 使用说明

容器镜像：flagos_earth_onecode:v1.0.0

## 快速开始

### 1. 下载模型包

```bash
# 默认下载到当前路径下AI-GAMFS文件夹，如需修改，则制定local_dir后的路径

modelscope download --model OneScience/AI-GAMFS --local_dir ./model
cd model
```

### 2. 使用方式
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

- 如果在科研工作中使用AI-GAMFS结果，建议引用AI-GAMFS原始论文和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。

```bibtex
@article{gui2026advancing,
  title={Advancing operational global aerosol forecasting with machine learning},
  author={Ke Gui, Xutao Zhang, Huizheng Che, Lei Li, Yu Zheng, Linchang An, Yucong Miao, Hujia Zhao, Oleg Dubovik, Brent Holben, Jun Wang, Pawan Gupta, Elena S. Lind, Carlos Toledano, Hong Wang, Zhili Wang, Yaqiang Wang, Xiaomeng Huang, Kan Dai, Xiangao Xia, Xiaofeng Xu, and Xiaoye Zhang},
  journal={Nature},
  year={2026},
  url={https://www.nature.com/articles/s41586-026-10234-y}
}
```