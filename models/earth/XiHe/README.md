<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## 仓库介绍
- OneScience整理的面向高分辨率全球海洋预报的XiHe大模型标准运行包
- 提供自然语言编程的配置、训练、推理、评测、可视化、数据准备和预检功能


## OneScience官方信息

| 平台 |  OneScience 主仓库 | Skills 仓库 |
|---|---|---|
| Gitee |  https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 使用方法

- 智能体使用

    `点击最上方图片按钮即可体验`

- 人工使用
    - [注册scnet](https://www.scnet.cn/ui/mall/)（`已有请忽略`）--> [快速开发](https://www.scnet.cn/ui/aihub/image/easyscience2024/2066709128882294786)
    --> 创建实例 --> 使用JupyterLba --> 终端
    - 使用过程
    ``` bash
    wget https://modelscope.cn/models/biaoliu/resource/resolve/master/onescience.sh && bash onesciecne.sh
    onescience train xihe ## xihe模型训练
    onescience val xihe ## xihe模型验证
    onescience test xihe ## xihe模型测试

## 文件说明
| 路径 | 类型 | 作用                                                      |
|---|------|---------------------------------------------------------|
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和输出                             |
| `conf/config.yaml` | 配置文件 | 已改写为 1993-1999 年 CMEMS 数据划分                             |
| `scripts/prepare_runtime_data.py` | 数据准备脚本 | 将 `OneScience/CMEMS` 下载结果整理到 `./data/data/`，并提取统计量、生成掩码 |
| `scripts/preflight.py` | 预检脚本 | 检查配置、年份、HDF5 结构、变量、统计量和掩码                               |
| `train.py` | 训练脚本 | 训练 XiHe 并输出 checkpoint 和 loss                           |
| `inference.py` | 推理脚本 | 使用 `data/checkpoints/model_bak.pth` 对测试集推理              |
| `result.py` | 评测与可视化脚本 | 计算 RMSE/ACC 并生成图片                                       |
| `work_slurm.sh` | 集群脚本 | SLURM/DCU 训练入口                                          |
| `model_run_skill.md` | skill文件 | 面向自然语言编程的大模型skill                                  |



## Note

* 当前标准包只适配`OneScience/CMEMS`中1993-1999年的7个HDF5文件。
* 仓库不包含预训练权重，推理前需要先训练或提供兼容checkpoint。

## 引用与许可证
* 模型：遵循Apache 2.0，可免费用于学术研究和商业用途
* 数据：请以 `OneScience/CMEMS` 数据集仓库说明为准
