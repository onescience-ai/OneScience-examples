<p align="center">
  <strong>
    <span style="font-size: 30px;">intent-obfuscator</span>
  </strong>
</p>

# 模型介绍

IntentObfuscator 是 arXiv:2405.03654《Can LLMs Deeply Detect Complex Malicious Queries? A Framework for Jailbreaking via Obfuscating Intent》的复现实现，面向 LLM 安全红队评估。框架通过混淆提示中的恶意意图（不修改恶意文本的 OI 方法，或增强歧义的 CA 方法）绕过 LLM 内容安全机制，并提供 REJ / ASR / HAL 三类攻击评估指标。

论文：Can LLMs Deeply Detect Complex Malicious Queries? A Framework for Jailbreaking via Obfuscating Intent
https://arxiv.org/abs/2405.03654

# 模型描述

- OI（Obscure Intention）：不修改恶意文本，使用遗传算法（算法1/2，Eq.23-24）生成混淆的正常意图模板并嵌入恶意问题。
- CA（Create Ambiguity）：用 LLM 将恶意问题改写为多重歧义句，再嵌入正常意图模板（算法3）。
- 评估：攻击成功判定满足三条件（Eq.1-2），统计 REJ / ASR / HAL（Section 6.1.5）。
- 主要模型文件：model/pipelines.py（端到端管线）、model/ga.py（遗传算法）、model/ca.py（歧义改写）、model/judge.py（攻击判定）、model/evaluate.py（指标统计）。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 红队安全评估 | 使用 HBP（Harmful Behavior Problems）数据集评估目标 LLM 对混淆恶意查询的越狱成功率 |
| 框架复现研究 | 复现 IntentObfuscator 的 OI / CA / Baseline 三条端到端管线 |
| 攻击成功判定 | 基于拒绝词 + 危险内容关键词自动判定成功/拒绝/幻觉 |
| 本地快速验证 | 使用 mock 后端快速检查数据读取、GA/CA 生成与指标计算 |

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

```bash
modelscope download --model OneScience/intent-obfuscator --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem、bio、all（全领域）
pip install onescience[all-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai 
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem（全领域）
pip install onescience[all-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

HBP（Harmful Behavior Problems）数据集，来自 llm-attacks/advbench 公开 CSV（约 520 条恶意指令文本），Tier 1 复现使用固定 seed=42 采样 20 条子集。数据位于仓库 `.paper2code_work/2405.03654/harmful_behaviors.csv`。

```bash
# 数据为公开 GitHub 资源，可自行下载：
# https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv
```

### 训练

本复现无传统梯度训练；"训练"阶段转化为 OI 模板生成（GA）与 CA 歧义改写 + 端到端评估。本地默认使用 Qwen2.5-0.5B-Instruct 作为目标 LLM。

单卡：

```bash
python scripts/run_all.py --config conf/tier1.yaml
```

冒烟验证（mock 后端，无需 GPU）：

```bash
python scripts/test_smoke.py
```

### 训练权重

本框架为黑盒攻击框架，无模型权重；目标 LLM 为本地 Qwen2.5-0.5B-Instruct（推理时按配置指定路径）。weight/ 目录暂为空，计划后续补充评估明细与基线数据。

### 推理

<!-- 推理脚本未在 scripts/ 目录中找到（无 infer/predict 命名脚本），OI/CA 端到端评测可运行 scripts/run_all.py -->

### 评估和可视化

```bash
python scripts/run_all.py --config conf/tier1.yaml
python scripts/evaluate.py
```

评估输出 REJ / ASR / HAL 统计报告至 `outputs/evaluation_report.json` 与 `.md`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 arXiv:2405.03654 原始论文的复现版本，用于学术研究与合规红队安全评估。

- 论文：Can LLMs Deeply Detect Complex Malicious Queries? A Framework for Jailbreaking via Obfuscating Intent
- arXiv: https://arxiv.org/abs/2405.03654

