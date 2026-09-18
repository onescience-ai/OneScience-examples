<p align="center">
  <strong>
    <span style="font-size: 30px;">PXDesign</span>
  </strong>
</p>

# 模型介绍

PXDesign 是字节跳动团队开源的蛋白质结合物（protein binder）从头设计模型套件，面向给定靶蛋白结构生成候选结合蛋白，并通过结构预测与置信度评估流程进一步筛选候选结构。

PXDesign 的完整工作流由 PXDesign 扩散生成模型、ProteinMPNN 序列设计、AF2-IG 评估以及 Protenix 评估等模块组成。官方提供 generation-only、preview 和 extended 三种主要运行方式，可用于从快速验证到完整候选筛选的不同场景。

论文：
> **PXDesign: Fast, Modular, and Accurate De Novo Design of Protein Binders**  
> https://www.biorxiv.org/content/10.1101/2025.08.15.670647v1

# 模型描述

PXDesign 的核心任务是根据目标蛋白结构及指定的设计区域生成新的蛋白质结合物。

典型流程如下：
```text
目标蛋白结构与设计约束
        -> PXDesign-d 扩散模型
        -> Binder Backbone Generation
        -> ProteinMPNN 序列设计
        -> AF2-IG 结构预测与过滤
        -> Protenix 结构预测与过滤（extended 模式）
        -> summary.csv
        -> 筛选高置信度 Binder
```

其中：
- **PXDesign-d**：根据目标蛋白结构、hotspot、binder 长度等条件生成候选 binder 骨架；
- **ProteinMPNN**：为生成的蛋白骨架设计氨基酸序列；
- **AF2-IG**：对候选 binder-target 复合物进行结构预测和质量过滤；
- **Protenix**：在 extended 模式下提供额外的结构预测与置信度评估；
- **summary.csv**：汇总候选结构的 AF2-IG、Protenix 等评价指标以及各过滤器的通过状态。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 蛋白质 Binder 从头设计 | 根据给定靶蛋白结构生成新的候选结合蛋白 |
| 界面定向设计 | 通过 hotspot 指定希望 binder 优先结合的目标残基 |
| 蛋白设计方案快速验证 | 使用 preview 模式快速评估设计任务及参数是否合理 |
| 高质量候选筛选 | 使用 extended 模式结合 AF2-IG 与 Protenix 进行多阶段过滤 |
| 结构生成研究 | 使用 `pxdesign infer` 仅执行 PXDesign 生成阶段 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：
[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

### 硬件要求

- PXDesign backbone generation 阶段建议使用 GPU/DCU；完整推理通常需要较大显存。
- MSA 生成与准备阶段主要使用 CPU，可提前通过 `prepare-msa` 或预计算 MSA 完成。
- ProteinMPNN、AF2-IG 与 Protenix 结合预测/筛选阶段依赖 PyTorch、JAX 等深度学习框架，推荐使用 GPU/DCU 运行。
- 如果 GPU/DCU 资源有限，可先用 CPU 单独完成 MSA 准备，再运行 PXDesign generation、ProteinMPNN、AF2-IG 与 Protenix 评估阶段。

### 安装运行环境

#### DCU 环境

```bash
# 请首先激活 DTK 和 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

# 支持 uv 安装
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

#### 环境说明

- 在实际运行过程中，如果遇到缺少依赖或版本问题，可根据 `requirements.txt` 声明的依赖版本额外安装依赖。
- 进入项目根目录并激活环境：

```bash
cd /path/to/PXDesign-main
conda activate your_env
```

将 PXDesign 安装到当前环境：

```bash
python -m pip install -e model
```

安装完成后验证：

```bash
which pxdesign
pxdesign --help
pxdesign pipeline --help
```

### 环境变量

建议进入 PXDesign 根目录后统一配置：
```bash
export PXDESIGN_ROOT=$PWD
export TOOL_WEIGHTS_ROOT=$PWD/weight/tool_weights
export PROTENIX_DATA_ROOT_DIR=$PWD/weight/release_data/ccd_cache
```

可通过：
```bash
echo $PXDESIGN_ROOT
echo $TOOL_WEIGHTS_ROOT
echo $PROTENIX_DATA_ROOT_DIR
```

确认配置。

## 权重与数据准备

PXDesign 完整流程依赖 PXDesign / Protenix 模型权重，以及 AlphaFold2、ProteinMPNN 和 CCD cache。本模型仓库已集成ccd_cache以及PXDesign/Protenix Checkpoint ，用户只需额外准备 `tool_weights/`部分即可。完整下载过程如下：

### 1）外部工具权重与 CCD Cache

PXDesign 官方提供下载脚本：
```bash
bash scripts/download_tool_weights.sh
```

该脚本沿用官方默认目录，直接运行时会在当前目录生成 `tool_weights/` 和 `release_data/ccd_cache/`。本项目已重构为 `weight/` 目录结构，因此推荐将已有权重和缓存整理或软链接到下述位置。

在本项目当前重构结构中，推荐将外部工具权重放置为：
```text
weight/
├── tool_weights/
│   ├── af2/        # AlphaFold2 权重
│   └── mpnn/       # ProteinMPNN 权重
└── release_data/
    └── ccd_cache/  # Protenix CCD cache
```

- CCD cache 默认推荐保存到：

```text
weight/release_data/ccd_cache/
```

如果需要指定其他位置，可设置：

```bash
export PROTENIX_DATA_ROOT_DIR=/path/to/ccd_cache
```

### 2）PXDesign 与 Protenix Checkpoint

以下模型权重会在首次运行时按需自动下载，也可以提前下载至对应位置：
```text
PXDesign diffusion checkpoint

Protenix checkpoints:
├── base
├── mini
└── mini_tmpl
```

当前重构结构中的推荐保存位置：

```text
weight/release_data/checkpoint/
```

所需文件包括：
```text
pxdesign_v0.1.0.pt
protenix_base_default_v0.5.0.pt
protenix_mini_default_v0.5.0.pt
protenix_mini_tmpl_v0.5.0.pt
```

### 3）检查

准备完成后可执行：
```bash
ls weight/tool_weights/af2/
ls weight/tool_weights/mpnn/
ls weight/release_data/ccd_cache/
ls weight/release_data/checkpoint/*.pt
```
确认所需权重与数据已经就绪。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/PXDesign --local_dir ./PXDesign
cd PXDesign
```

- PXDesign 额外依赖 Protenix 和 PXDesignBench，本模型仓库已集成对应依赖源码，无需单独下载。
- PXDesign 完整流程还依赖 AlphaFold2、ProteinMPNN 以及 Protenix 所需的 CCD cache，请先按照“权重与数据准备”完成相关资源准备。

### 快速验证

首先验证命令是否可用：
```bash
pxdesign --help
```

如需按照本文示例将运行结果输出到 `runs/`，先创建输出目录：
```bash
mkdir -p runs
```

然后检查官方示例 YAML：
```bash
pxdesign check-input \
  --yaml conf/examples/PDL1_quick_start.yaml
```

成功时应输出：
```text
YAML file is valid.
```

### 示例数据

当前项目提供：
```text
conf/examples/
├── PDL1_quick_start.yaml
├── 5o45.cif
└── msa/
    └── PDL1/
        └── 0/
```

其中 `PDL1_quick_start.yaml` 用于定义 PDL1 binder 设计任务。

典型 YAML 格式如下：
```yaml
target:
  file: "./conf/examples/5o45.cif"
  chains:
    A:
      crop: ["1-116"]
      hotspots: [40, 99, 107]
      msa: "./conf/examples/msa/PDL1/0"

binder_length: 80
```

主要字段：
| 字段 | 说明 |
| --- | --- |
| `target.file` | 目标蛋白结构文件，可使用 mmCIF 或 PDB |
| `target.chains` | 参与设计的目标链 |
| `crop` | 从目标链中保留的残基范围 |
| `hotspots` | 用于引导 binder 界面生成的目标残基 |
| `msa` | 目标链预计算 MSA 路径 |
| `binder_length` | 待设计 binder 的氨基酸长度 |

PXDesign 内部主要使用 mmCIF 的 `label_seq_id` 作为标准残基索引。对于自定义任务，建议优先使用 mmCIF 文件，并通过 `parse-target` 检查 crop 与 hotspot 是否指向预期位置。

### 输入检查与目标解析

#### 1）检查 YAML

正式执行设计任务前建议首先运行：

```bash
pxdesign check-input \
  --yaml conf/examples/PDL1_quick_start.yaml
```

#### 2）解析目标并生成可视化调试文件
```bash
pxdesign parse-target \
  --yaml conf/examples/PDL1_quick_start.yaml \
  -o runs/debug_target
```

该步骤适合在正式进行大规模设计之前检查：

- crop 是否正确；
- hotspot 是否对应预期残基；
- 结构链与残基编号是否正确。

## 推理示例

PXDesign 官方主要提供三种运行模式：
```text
Generation Only
    -> 只生成 PXDesign binder backbone

Preview Pipeline
    -> PXDesign + ProteinMPNN + AF2-IG

Extended Pipeline
    -> PXDesign + ProteinMPNN + AF2-IG + Protenix
```

### 1. Generation Only：仅运行 PXDesign 生成

#### 快速 smoke test

为了先验证模型、权重和 GPU/DCU 是否可以工作，可使用较小步数：
```bash
pxdesign infer \
  -i conf/examples/PDL1_quick_start.yaml \
  -o runs/test_infer \
  --load_checkpoint_dir weight/release_data/checkpoint \
  --N_sample 1 \
  --N_step 20 \
  --dtype bf16 \
  --sample_diffusion_chunk_size 1
```

#### 完整步数生成测试

```bash
pxdesign infer \
  -i conf/examples/PDL1_quick_start.yaml \
  -o runs/test_infer_full \
  --load_checkpoint_dir weight/release_data/checkpoint \
  --N_sample 10 \
  --N_step 400 \
  --dtype bf16
```

该模式只执行 binder 生成，不提供完整 AF2 / Protenix 过滤结果。

### 2. Preview Pipeline

Preview 模式执行：
```text
PXDesign generation
        -> ProteinMPNN sequence design
        -> AF2-IG filtering
```

```bash
pxdesign pipeline \
  --preset preview \
  -i conf/examples/PDL1_quick_start.yaml \
  -o runs/test_preview \
  --load_checkpoint_dir weight/release_data/checkpoint \
  --N_sample 2 \
  --N_step 100 \
  --dtype bf16 \
  --use_fast_ln False \
  --use_deepspeed_evo_attention False
```

Preview 模式适合：
- 首次验证完整 pipeline；
- 检查 hotspot / crop 是否合理；
- 判断当前设计任务难度；
- 在进行大规模 Extended 任务前做小规模预实验。

### 3. Extended Pipeline

Extended 模式是 PXDesign 官方用于完整评价的流程：

```text
PXDesign generation
        -> ProteinMPNN
        -> AF2-IG
        -> Protenix
        -> summary.csv
```

#### 小规模验证
```bash
pxdesign pipeline \
  --preset extended \
  -i conf/examples/PDL1_quick_start.yaml \
  -o runs/test_extended \
  --load_checkpoint_dir weight/release_data/checkpoint \
  --N_sample 2 \
  --N_step 100 \
  --dtype bf16 \
  --use_fast_ln False \
  --use_deepspeed_evo_attention False
```

#### Quick Start 规模

官方 Quick Start 示例使用：
```text
N_sample = 10
N_step = 400
```

```bash
pxdesign pipeline \
  --preset extended \
  -i conf/examples/PDL1_quick_start.yaml \
  -o runs/test_extended_N10 \
  --load_checkpoint_dir weight/release_data/checkpoint \
  --N_sample 10 \
  --N_step 400 \
  --dtype bf16 \
  --use_fast_ln False \
  --use_deepspeed_evo_attention False
```

## 输出说明

Extended 模式的核心结果通常位于：
```text
<OUT_DIR>/
└── design_outputs/
    └── <task_name>/
        ├── summary.csv
        ├── task_info.json
        ├── server_extended_mode.png
        ├── orig_designed/
        ├── passing-AF2-IG-easy/
        └── passing-Protenix-basic/
```

如果没有设计通过相应过滤器，`passing-AF2-IG-easy/` 或 `passing-Protenix-basic/` 目录可能不会生成，这是小样本测试中的正常现象。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |


# 引用与许可证

如果在科研工作中使用 PXDesign，建议引用 PXDesign 官方论文：
```bibtex
@article{ren2025pxdesign,
  title={PXDesign: Fast, Modular, and Accurate De Novo Design of Protein Binders},
  author={Ren, Milong and Sun, Jinyuan and Guan, Jiaqi and Liu, Cong and
          Gong, Chengyue and Wang, Yuzhe and Wang, Lan and Cai, Qixu and
          Chen, Xinshi and Xiao, Wenzhi},
  journal={bioRxiv},
  pages={2025--08},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

PXDesign 完整 pipeline 还依赖 Protenix、ProteinMPNN 和 AF2-IG 等方法。若在科研工作中实际使用这些模块，建议同时按照 PXDesign 官方 README 中的说明引用对应原始工作。

Protenix：
```bibtex
@article{bytedance2025protenix,
  title={Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  author={ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan
          and Lu, Chan and Ma, Wenzhi and Guan, Jiaqi and Gong, Chengyue
          and Yang, Jincai and Zhang, Hanyu and Zhang, Ke and Wu, Shenghao
          and Zhou, Kuangqi and Yang, Yanping and Liu, Zhenyu and Wang, Lan
          and Shi, Bo and Shi, Shaochen and Xiao, Wenzhi},
  year={2025},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.01.08.631967}
}
```

ProteinMPNN：
```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel
          and Bai, Hua and Ragotte, Robert J and Milles, Lukas and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022}
}
```

AF2-IG：
```bibtex
@article{bennett2023improving,
  title={Improving de novo protein binder design with deep learning},
  author={Bennett, Nathaniel R and Coventry, Brian and Goreshnik, Inna
          and Huang, Buwei and Allen, Aza and Vafeados, Dionne and others},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={2625},
  year={2023}
}
```

PXDesign 官方仓库采用 **Apache License 2.0**。根据官方 README，该许可证允许学术研究和商业使用。使用、修改或再分发代码时应遵守本项目 `LICENSE` 中的具体条款。

此外：
- AlphaFold2 / AF2 权重及相关资源应遵守其对应许可证和使用条款；
- ProteinMPNN 应遵守其官方仓库许可证；
- Protenix 应遵守其官方仓库许可证；
- 通过 SCNet 共享目录复用的模型和数据资源仍应遵循原始资源各自的授权条件。

如用于论文、报告或公开发布，建议同时引用 PXDesign、Protenix、ProteinMPNN、AF2-IG 以及实际使用的其他第三方模型和数据资源。
