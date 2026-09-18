<p align="center"><strong><span style="font-size: 30px;">dl_binder_design</span></strong></p>

# 模型介绍

`dl_binder_design` 是 Bennett 等人提出的从头蛋白质结合剂设计流程。它接收 binder–target 复合物骨架，使用 ProteinMPNN 设计 binder 序列，通过 PyRosetta FastRelax 优化结构，再以带 initial guess 修改的 AlphaFold2 预测复合物并计算置信度指标。

- 论文：[Improving de novo protein binder design with deep learning](https://www.nature.com/articles/s41467-023-38328-5)

# 模型描述

本模型包不是单一神经网络，而是由多个预训练模型与结构计算工具组成的复合推理流程：

```text
binder–target 复合物骨架（PDB / Rosetta silent）
        │
        ├─ ProteinMPNN：设计 binder 氨基酸序列
        ├─ PyRosetta FastRelax：优化复合物结构（可选）
        └─ AlphaFold2 initial guess：复核结构并输出 pLDDT、PAE 与 RMSD
```

本流程不从 target 单独生成 binder 骨架。输入骨架可由 RFdiffusion 等工具预先生成，也可直接使用模型包中的示例。



# 适用场景

| 场景 | 说明 |
| --- | --- |
| Binder 序列设计 | 在给定 binder–target 骨架时生成 binder 候选序列 |
| 序列/骨架联合优化 | 交替执行 ProteinMPNN 与 FastRelax |
| 复合物结构复核 | 使用 AlphaFold2 initial guess 重新预测设计结构 |
| 候选筛选 | 按 pLDDT、PAE、`pae_interaction` 和对齐 RMSD 排序 |
| 批量任务 | 支持 PDB 目录、runlist、checkpoint 和 silent file |

# 使用说明

## 1. OneCode 使用

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装

**硬件要求**


- ProteinMPNN 支持 CPU 和 PyTorch 可见的加速设备；
- PyRosetta FastRelax 主要使用 CPU，建议使用性能较好的多核 CPU；运行时间取决于结构长度、设计数量和 `relax_cycles`；

- AF2 的内存和运行时间会随复合物长度及 `recycle` 数增加；
- 磁盘需能够容纳约 356 MB 的 `params_model_1_ptm.npz`、ProteinMPNN 权重、PyRosetta 安装与数据库，以及推理产生的 PDB、checkpoint、评分和临时文件。
### 下载模型包

安装 ModelScope 命令行工具后下载模型：

```bash
python -m pip install modelscope
modelscope download --model OneScience/dl_binder_design --local_dir ./dl_binder_design
cd dl_binder_design
```


### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```
随后安装其额外通用依赖：

```bash
python -m pip install --no-deps -r requirements.txt
```


## 4. 输入要求

- 第一条链为待设计的 binder，第二条链为保持固定的 target；
- 当前 AF2 脚本最多处理两条链；
- 建议不同链使用不重叠的残基编号；脚本检测到重复编号后默认会重新编号；
- 输入应包含合理的主链原子和清晰的链边界；
- binder 中需要保持不变的位置可使用 PDB residue label `FIXED` 标记。

示例位于 `conf/examples/inputs/`；包括 10 个 PDB 复合物骨架、ProteinMPNN 参考输出以及 Rosetta silent 文件。

## 5. 环境自检

```bash
python model/include/importtests/proteinmpnn_importtest.py
python model/include/importtests/af2_importtest.py
```



## 6. 统一推理入口

仅运行 ProteinMPNN，不做 FastRelax：

**作用：** 快速为输入骨架设计 binder 序列，适合验证 ProteinMPNN 权重加载和批量生成候选序列。

```bash
python scripts/inference.py mpnn \
  --input-dir conf/examples/inputs/pdbs \
  --output-dir output/mpnn_no_relax \
  --relax-cycles 0 \
  --seqs-per-struct 1 \
  --debug
```

运行 ProteinMPNN + 1 次 FastRelax：

**作用：** 在序列设计后进行结构松弛，减少局部结构冲突并优化复合物骨架。

```bash
python scripts/inference.py mpnn \
  --input-dir conf/examples/inputs/pdbs \
  --output-dir output/mpnn_fastrelax \
  --relax-cycles 1 \
  --seqs-per-struct 1 \
  --debug
```

AF2 CPU 单样本 smoke test：

**作用：** 使用 AF2 initial guess 复核单个设计结构，并输出 pLDDT、PAE 和 RMSD 等置信度指标。

```bash
mkdir -p output
printf '%s\n' 'design_ppi_0_dldesign_0' > output/af2_smoke.list

python scripts/inference.py af2 \
  --input-dir conf/examples/inputs/proteinmpnn_output_pdbs \
  --output-dir output/af2_cpu \
  --runlist output/af2_smoke.list \
  --recycle 1 \
  --debug
```

串联执行完整 PDB 目录流程：

**作用：** 依次完成序列设计、FastRelax 和 AF2 复核，用于端到端批量生成与评估候选结合剂。

```bash
python scripts/inference.py pipeline \
  --input-dir conf/examples/inputs/pdbs \
  --output-dir output/pipeline \
  --relax-cycles 1 \
  --recycle 1 \
  --debug
```

输出包括设计/预测 PDB、checkpoint 与 AF2 `.sc` 评分文件。`pae_interaction` 越低通常表示 binder–target 界面置信度越高，但该指标不能替代实验结合验证。

## 7. Silent file 与 FIXED 位点

统一入口面向 PDB 目录；silent file 可调用底层脚本：

- ProteinMPNN silent 推理用于直接批量设计 silent 文件中的结构，并将结果集中写入新的 silent 文件；
- AF2 silent 推理用于复核设计后的 silent 结构并生成预测结果和评分。

```bash
python model/mpnn_fr/dl_interface_design.py \
  -silent conf/examples/inputs/in.silent \
  -outsilent output/mpnn_fastrelax.silent

python model/af2_initial_guess/predict.py \
  -silent output/mpnn_fastrelax.silent \
  -outsilent output/af2.silent \
  -scorefilename output/af2.sc
```

如 RFdiffusion 同时提供 `.trb` 文件，可写入固定残基标签：

```bash
python model/helper_scripts/addFIXEDlabels.py \
  --pdbdir /path/to/pdbs \
  --trbdir /path/to/trbs \
  --verbose
```



## 训练

`dl_binder_design` 是调用 ProteinMPNN、PyRosetta 和 AlphaFold2 官方预训练权重的推理编排流程，本仓库没有独立训练目标、项目级训练入口或配套训练数据集。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[Improving de novo protein binder design with deep learning](https://www.nature.com/articles/s41467-023-38328-5)。
- 顶层流程代码来自 [nrbennet/dl_binder_design](https://github.com/nrbennet/dl_binder_design)，采用 MIT License。
- ProteinMPNN 代码来自 [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)，采用 MIT License。
- AlphaFold2 源码采用 Apache License 2.0；[模型参数采用 CC BY 4.0](https://github.com/google-deepmind/alphafold#model-parameters)，发布时须保留归属信息并遵守上游条款。
- PyRosetta 具有独立许可要求，商业用途需要另行取得许可。
- 其他第三方代码、数据和权重分别受其原始版权声明、许可证和使用条款约束。
