<p align="center">
  <strong>
    <span style="font-size: 30px;">Flex ddG</span>
  </strong>
</p>

# 模型介绍

Flex ddG 是基于 Rosetta 的蛋白质-蛋白质界面突变效应预测流程，用于估计突变前后结合自由能变化（interface ΔΔG）。该方法利用 Rosetta Backrub 协议采样局部骨架构象，再对野生型与突变体进行侧链重排、结构最小化和界面能量计算，从而评估突变对蛋白质结合亲和力的影响。

论文：

> **Flex ddG: Rosetta Ensemble-Based Estimation of Changes in Protein–Protein Binding Affinity upon Mutation**  
> https://doi.org/10.1021/acs.jpcb.7b11367

# 模型描述

Flex ddG 不是依赖神经网络权重的模型，而是一套基于 Rosetta 能量函数和构象采样的计算流程。输入通常包括蛋白质复合物 PDB、界面链信息以及描述突变的 Rosetta resfile。流程通过 Backrub 采样生成构象集合，对野生型和突变体分别优化并计算界面能量，最终得到 ΔΔG。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 蛋白质-蛋白质界面突变效应预测 | 预测突变造成的结合自由能变化 ΔΔG |
| 界面热点残基分析 | 评估特定位点突变对结合稳定性的影响 |
| 单点饱和突变扫描 | 对目标位点生成 20 种标准氨基酸替换并分别计算 ΔΔG |
| 蛋白质工程与界面优化 | 辅助筛选可能增强或削弱蛋白质相互作用的突变 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- Flex ddG 的核心计算由 Rosetta CPU 程序完成，标准流程不依赖 GPU/DCU。
- 官方 Python 脚本使用 `multiprocessing` 并发启动多个 Rosetta 实例。每个 Rosetta 实例大约需要 2 GB 内存，应根据 CPU 核数和节点内存设置并发数。

### 安装运行环境

#### DCU环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

# 支持uv安装
pip install onescience[bio] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

#### 环境说明

- Flex ddG 的核心依赖是 **Rosetta**。主流程需要 Rosetta 编译生成的 `rosetta_scripts` 可执行程序。


### Rosetta 安装

Flex ddG 不需要神经网络模型权重以及额外下载大型数据集，但必须单独准备 Rosetta。

#### 1）安装 Rosetta

Rosetta 需要按照 RosettaCommons 官方说明单独申请许可并下载安装：

```text
https://www.rosettacommons.org/software
```

Rosetta 的许可与 Flex ddG 教程仓库的 MIT License 相互独立。学术和非商业用户可申请非商业许可证，商业使用需另行取得许可。

安装后请确认至少存在：

```text
/path/to/rosetta/source/bin/rosetta_scripts
/path/to/rosetta/source/bin/score_jd2
```

#### 2）配置 Rosetta 路径

运行前修改 `scripts/run_example_1.py` 与 `scripts/run_example_2_saturation.py` 中：

```python
rosetta_scripts_path = os.path.expanduser("~/rosetta/source/bin/rosetta_scripts")
```

如需执行 `scripts/extract_structures.py`，还需要将脚本中的 `score_jd2_path` 修改为实际 Rosetta `score_jd2` 路径。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/flex_ddG --local_dir ./flex_ddG
cd flex_ddG
```

- Flex ddG **额外依赖 Rosetta**，本模型仓库不内置 Rosetta；请先按照“Rosetta 安装”完成 Rosetta 的许可申请与安装，并配置 `rosetta_scripts_path`。
- 如果需要从 `struct.db3` 提取 PDB 结构，还需要配置 Rosetta 的 `score_jd2` 路径。
- 本文默认参数仅用于快速功能验证，不代表正式科研计算参数。

### 快速验证

首先确认 Rosetta 可执行程序可用：

```bash
/path/to/rosetta/source/bin/rosetta_scripts -help
```

运行官方示例：

```bash
python scripts/run_example_1.py
```

成功后会生成：

```text
output/
```

# 示例数据

官方示例目录为：

```text
scripts/inputs/
└── 1JTG/
    ├── 1JTG_AB.pdb
    ├── chains_to_move.txt
    ├── nataa_mutations.resfile
    ├── mutations.resfile
    ├── mutations.mutfile
    ├── pdb2rosetta.resmap.json
    └── rosetta2pdb.resmap.json
```

其中：

| 文件 | 说明 |
| --- | --- |
| `1JTG_AB.pdb` | 蛋白质复合物结构 |
| `chains_to_move.txt` | 定义界面计算中作为一侧移动的链 |
| `nataa_mutations.resfile` | Flex ddG 实际使用的突变 resfile |
| `mutations.resfile` | 示例突变配置 |
| `mutations.mutfile` | 示例突变信息 |

Flex ddG 使用的 resfile 需要以 `NATAA` 开头。官方脚本明确说明，不应以 `NATRO` 替代，否则会改变突变体附近残基的重排行为并造成 ΔΔG 偏差。

对于自己的任务，通常至少需要：

```text
复合物 PDB
+ 界面链信息
+ 描述突变的 resfile
```

# 推理示例

## 指定突变的 Flex ddG 计算

确认 `scripts/run_example_1.py` 中 Rosetta 路径设置正确：

```python
rosetta_scripts_path = "/path/to/rosetta/source/bin/rosetta_scripts"
```

运行：

```bash
python scripts/run_example_1.py
```

脚本会读取 `scripts/inputs/` 中的复合物结构、`chains_to_move.txt` 与 `nataa_mutations.resfile`，并调用 `conf/ddG-backrub.xml` 执行 Flex ddG。

主要参数：

| 参数 | 默认值 | 正式计算常用设置/含义 |
| --- | ---: | --- |
| `nstruct` | 3 | 正式使用通常约 35 或更多独立重复 |
| `number_backrub_trials` | 10 | 官方 benchmark 常用 35000 |
| `max_minimization_iter` | 5 | 正常值 5000 |
| `abs_score_convergence_thresh` | 200.0 | 正常值 1.0 |
| `backrub_trajectory_stride` | 5 | 控制 Backrub 轨迹检查点间隔 |

本文的小参数用于缩短运行时间，不能直接作为正式 ΔΔG 计算参数。

## 单点饱和突变

运行：

```bash
python scripts/run_example_2_saturation.py
```

脚本会对指定残基依次生成 20 种标准氨基酸突变对应的 resfile 并运行 Flex ddG。

目标位点在脚本中配置：

```python
residue_to_mutate = ('B', 49, '')
```

格式为：

```text
(链 ID, PDB 残基编号, insertion code)
```

结果保存在：

```text
output_saturation/
```

## 并行运行建议

官方脚本默认：

```python
use_multiprocessing = True
max_cpus = 2
```

用户在实际运行过程中，可根据申请的 CPU 核数和节点内存调整 `max_cpus`。由于每个 Rosetta 实例都会独立占用 CPU 和内存，不建议无条件设置为全部核心。

# 结果分析

Example 1 完成后：

```bash
python scripts/analyze_flex_ddG.py output
```

饱和突变结果：

```bash
python scripts/analyze_flex_ddG.py output_saturation
```

分析脚本会输出：

```text
wt_dG
mut_dG
ΔΔG
```

并将结果写入：

```text
analysis_output/
```

中的 CSV 文件。突变 ΔΔG 还会根据原始 Flex ddG 论文拟合的 GAM 模型给出重加权结果。

如需提取 Backrub、野生型最小化或突变体最小化后的结构：

```bash
python scripts/extract_structures.py output
```

该脚本会查找 `struct.db3` 并调用 Rosetta `score_jd2` 导出 PDB。

# 输出说明

执行 `scripts/run_example_1.py` 后，主要结果位于：

```text
output/
└── <case>/
    └── <replicate>/
        ├── rosetta.out
        ├── ddG.db3
        └── struct.db3
```

其中：

| 文件 | 说明 |
| --- | --- |
| `rosetta.out` | Rosetta 运行日志 |
| `ddG.db3` | Flex ddG 能量与轨迹相关结果数据库 |
| `struct.db3` | Rosetta 生成结构数据库 |
| `analysis_output/*.csv` | `scripts/analyze_flex_ddG.py` 生成的 ΔΔG 汇总结果 |

分析脚本会从每个运行生成的 `ddG.db3` 读取 Backrub trajectory stride，因此通常不需要手工修改分析脚本。若数据库中没有 stride，可使用：

```bash
python scripts/analyze_flex_ddG.py output --stride N
```
进行覆盖。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Flex ddG 原始论文：[Flex ddG: Rosetta Ensemble-Based Estimation of Changes in Protein–Protein Binding Affinity upon Mutation](https://doi.org/10.1021/acs.jpcb.7b11367)。
- Flex ddG 使用 Rosetta Backrub 构象采样方法；相关论文：[Backrub-Like Backbone Simulation Recapitulates Natural Protein Conformational Variability and Improves Mutant Side-Chain Prediction](https://doi.org/10.1016/j.jmb.2008.05.023)。
- flex_ddG_tutorial 官方源码采用 MIT License，详见仓库根目录 `LICENSE`。
- **Rosetta 不属于 Flex ddG 教程仓库的 MIT License 范围。** Rosetta 使用独立的软件许可；学术和非商业用户可申请非商业许可证，商业使用需要另行取得商业许可。
- 科研使用时建议同时引用 Flex ddG、Backrub 与 Rosetta 相关文献，并按照 OneScience 相关项目要求补充引用。
