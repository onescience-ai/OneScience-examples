---
license: cc-by-nc-sa-4.0
language:
- en
- zh
tags:
- OneScience
- 地球科学
- 气象预报
- 多模态推理
- VLM
frameworks: PyTorch
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">Weather-R1</span>
  </strong>
</p>


# 模型介绍

Weather-R1 是首个在气象领域具有逻辑一致性的多模态推理视觉语言模型 (VLM)。它基于 Qwen2.5-VL-7B 构建，通过逻辑一致性强化学习 (LoCo-RFT) 进行微调，旨在解决主流强化学习中常见的“自相矛盾推理”问题。该模型在 WeatherQA 气象多模态基准测试中表现出色，相比基准模型准确率提升了 9.8 个百分点。


# 仓库说明

本仓库是 OneScience 整理的 Weather-R1 最小可运行模型仓库，已集成 `flag_gems` 算子库进行性能优化，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- **推理**：支持多模态气象问答推理，已适配海光 (Hygon) DCU 环境。
- **算子优化**：集成 `flag_gems`，针对 Transformer 架构进行算子级加速。
- **数据自动获取**：提供脚本支持从 Hugging Face 镜像站快速下载 WeatherQA 数据集。

当前不支持能力：

- **在线微调**：本仓库主要面向推理验证，全量训练需配合 EasyR1 环境。


# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 气象多模态问答 | 输入气象卫星图/形势图及问题，输出逻辑严密的推理过程及答案 |
| 算子加速验证 | 验证 `flag_gems` 在海光 DCU 等国产算力上的加速效果与兼容性 |


# 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `src/` | 模型核心推理逻辑代码 | 包含适配 `flag_gems` 的入口 |
| `scripts/` | 运行与评估辅助脚本 | - |
| `run.sh` | 一键式推理测试脚本 | 已封装环境清理与依赖安装 |
| `download_data.py` | WeatherQA 数据集下载脚本 | 支持断点续传与镜像加速 |
| `requirements/` | 环境依赖清单 | 包含 easyr1 和 vllm 版本的依赖 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 NVIDIA GPU 或 海光 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 FlagOS 镜像环境。

**环境清理 (重要)**

为了规避底层库冲突，建议在安装前清理无关框架：
```bash
pip uninstall -y tensorflow jax jaxlib vllm
```

**环境检测**

- 海光 DCU：

```bash
hy-smi
```

## 快速开始

### 1. 下载模型包

```bash
# 默认下载到当前路径下 model 文件夹
modelscope download --model OneScience/Weather-R1 --local_dir ./model
cd model
```

### 2. 安装运行环境与下载数据

```bash
# 安装必要依赖
pip install flag_gems transformers qwen-vl-utils tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 WeatherQA 测试数据集 (约 4.6GB)
python download_data.py
```

### 3. 使用方式 (一键测试)

```bash
# 赋予执行权限
chmod +x run.sh

# 运行推理测试 (参数: 模型ID/路径 数据JSON路径 图片目录路径)
./run.sh "Marco711/Weather-R1/LoCo-RFT/WeatherQA-Rain" "data/WeatherQA/all_split.json" "data/WeatherQA/image"
```

运行结束后，可在 `gems_debug.log` 中查看 `flag_gems` 的算子接管情况。


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |


# 引用与许可证

- Weather-R1 原始代码使用 Creative Commons Attribution Non Commercial Share Alike 4.0 许可证。
- 如果在科研工作中使用结果，请引用原始论文：

```bibtex
@misc{wu2026weatherr1logicallyconsistentreinforcement,
      title={Weather-R1: Logically Consistent Reinforcement Fine-Tuning for Multimodal Reasoning in Meteorology}, 
      author={Kaiyu Wu and Pucheng Han and Hualong Zhang and Naigeng Wu and Keze Wang},
      year={2026},
      eprint={2601.14044},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.14044}, 
}
```

