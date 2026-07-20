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
    <span style="font-size: 30px;">DDIM-DSC</span>
  </strong>
</p>


# 模型介绍

几句话介绍下模型，由谁开发、实现什么功能等。


# 仓库说明

本仓库是OneScience整理的{DDIM-DSC}最小可运行模型仓库，面向ModelScope下载、OneCode自动化运行和本地快速验证场景。

当前支持能力：

- 训练、推理、微调

当前不支持能力：

- 无

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 描述可用的场景 | 输入、输出、功能等 |
| 将低分辨率图像转换为高分辨率图像 | 输入为每组2通道（u和v分量）、大小为32×32的低分辨率风场数据，输出为每组2通道（u和v分量）、空间尺寸为32×32的生成风场数据以及评价指标 |
# 文件说明

| 路径 | 功能 |备注 |
| :---: | :---: | :---: |
| `README.md` |  工程使用说明文档 | 中文为主 |
| `config/sheduler/scheduler_config.json` | 配置文件，声明FASTA、输出目录、权重目录和数据集位置 | 已适配标准包相对路径 |
| `scripts/test.py` | 训练脚本 | 需指定数据路径 |
| `model/unet/unet_weights_download.py` | 权重文件  | 需提前训练得到权重 |
| `requirements.txt` | 依赖包  | DCU版依赖包 |


# 使用说明

## 1. OneCode使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用GPU或DCU运行。
- CPU可以用于连通性验证，但速度较慢。
- DCU用户需要预先安装DTK，建议使用DTK 25.04.2以上版本或与当前集群匹配的OneScience推荐版本。

**软件要求**

请参考requirements.txt，DCU用户想了解更多适配内容请联系 liubiao@sugon.com

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光DCU：

```bash
hy-smi
```

## 快速开始

### 1. 下载模型包

```bash
# 默认下载到当前路径下{model}文件夹，如需修改，则制定local_dir后的路径

modelscope download --model OneScience/{DDIM-DSC} --local_dir ./model
cd model
```

### 2. 安装运行环境

```bash
pip install -r requirements.txt
```

预检会检查配置文件、样例输入、权重文件和必要路径。该步骤不会加载完整3B模型，适合先确认模型包完整性。

### 4. 使用方式

```bash
# 训练
python scripts/test.py

# 评测和可视化
python scripts/test.py
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- {DDIM-DSC} 原始代码使用 MIT License。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

- 如果在科研工作中使用 {DDIM-DSC} 结果，建议引用 {DDIM-DSC} 原始论文和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。

