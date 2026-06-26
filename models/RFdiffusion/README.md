# RFdiffusion

<p align="center">
    <a href="https://www.modelscope.cn/studios/OneScience/OneScience" target="_blank">
        <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
    </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

RFdiffusion 是面向蛋白质结构生成与蛋白设计的扩散模型。它可以在无条件生成、基序支架设计、对称寡聚体设计、蛋白-蛋白相互作用设计和部分扩散设计等场景中生成候选蛋白质骨架，并输出 PDB 结构、扩散轨迹和运行元数据。

本仓库是 OneScience 标准化后的 RFdiffusion 可运行模型包，ModelScope ID 为 `OneScience/RFdiffusion`。包内保留 OneScience 示例目录中的脚本、配置、示例输入 PDB、辅助脚本和图片，并内置 `models/` 权重目录；用户下载后可以直接在包根目录运行预检，也可以使用 `scripts/run_inference.py` 按 Hydra 参数执行推理。

## Resource Card

| 字段 | 内容 |
| --- | --- |
| 资源类型 | model |
| OneScience 领域 | bio |
| 领域标签 | protein design, protein structure, diffusion |
| 任务 | protein_structure_design |
| 任务标签 | unconditional generation, motif scaffolding, binder design |
| ModelScope ID | `OneScience/RFdiffusion` |
| 主平台资源 | https://modelscope.cn/models/OneScience/RFdiffusion |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/biosciences/RFdiffusion` |
| 支持能力 | 预检、推理、示例输入验证 |
| 必需模型文件 | `models/Base_ckpt.pt`；其他 checkpoint 按任务选用 |
| 必需数据集 | 无强依赖训练数据；推理输入 PDB 可由用户提供或使用 `examples/input_pdbs/` |
| 最小验证 | `python tools/preflight_check.py --repo-root .` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `README.md` | 说明文件 | 人类和大模型读取入口 | 是 | 全部能力 | 仓库根目录 | 本文件 |
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和诊断信息 | 是 | 全部能力 | 仓库根目录 | 修改运行包时必须同步更新 |
| `onescience_relations.yaml` | 关系索引 | 声明模型与外部输入或数据集的关系 | 是 | 自动解析 | 仓库根目录 | 当前无必需数据集 |
| `scripts/run_inference.py` | 推理脚本 | RFdiffusion Hydra 推理入口 | 是 | 推理 | 仓库根目录下相同路径 | 可用 Hydra 参数覆盖配置 |
| `config/inference/base.yaml` | 配置文件 | 基础推理配置和默认权重目录约定 | 是 | 推理 | 仓库根目录下相同路径 | 标准命令覆盖 `inference.model_directory_path=$PWD/models` |
| `config/inference/symmetry.yaml` | 配置文件 | 对称设计相关配置 | 否 | 对称推理 | 仓库根目录下相同路径 | 对称设计示例使用 |
| `models/` | 模型权重目录 | RFdiffusion checkpoint 和结构预测权重 | 是 | 推理、任务切换 | 仓库根目录下相同路径 | 总大小约 3.9G |
| `examples/` | 示例脚本和输入 | 官方示例命令、PDB 输入、支架文件和目标折叠条件文件 | 是 | 推理、最小样例 | 仓库根目录下相同路径 | 用户也可替换为自己的 PDB |
| `helper_scripts/` | 辅助脚本 | 二级结构和邻接矩阵等辅助处理 | 否 | 高级示例 | 仓库根目录下相同路径 | 部分支架设计流程使用 |
| `img/` | 图片资源 | 原始说明文档配图 | 否 | 文档 | 仓库根目录下相同路径 | 便于阅读 |
| `RFdiffusion_README.md` | 原始说明 | OneScience 示例目录原说明文档 | 否 | 文档 | 仓库根目录 | 保留原有示例说明 |
| `tools/preflight_check.py` | 预检脚本 | 检查 Manifest、README 编码、关键文件、权重和 repo_id 一致性 | 是 | 预检 | 仓库根目录下相同路径 | 上传前后均使用 |
| `checksums.sha256` | 校验清单 | 标准包文件 SHA256 | 是 | 校验 | 仓库根目录 | 上传前生成 |

## Manifest

Manifest 文件位于仓库根目录 `manifest.yaml`。修改仓库 ID、下载命令、运行命令、文件路径、权重目录或输入关系后，必须同步更新 Manifest，并执行 `python tools/preflight_check.py --repo-root .` 验证 YAML 可解析、关键字段存在、repo_id 一致、README 无乱码。

## 模型 vs 数据集关系

RFdiffusion 推理没有强制依赖固定训练数据集，因此 `relations.required_datasets` 为空。推理输入可以由用户提供 PDB 文件，也可以使用仓库内置的 `examples/input_pdbs/` 示例 PDB；Manifest 中通过 `relations.optional_inputs` 声明这类输入文件。训练数据不随本模型包发布。

## 文件与下载

使用 ModelScope CLI 下载模型包：

```bash
modelscope download --model OneScience/RFdiffusion --local_dir ./RFdiffusion
```

如网页端使用 `--cache_dir` 下载，运行前应切换到实际下载后的模型包根目录。所有 README、Manifest、下载命令和关系索引中的 repo_id 均为 `OneScience/RFdiffusion`。

## 环境安装

推荐在已部署的 OneScience 生物信息环境中运行。最小预检只需要 Python 和 PyYAML；实际推理需要 OneScience、PyTorch、Hydra/OmegaConf 以及 RFdiffusion 相关依赖。

```bash
cd ./RFdiffusion
python tools/preflight_check.py --repo-root .
```

## 运行流程

### 1. 环境预检

```bash
cd ./RFdiffusion
python tools/preflight_check.py --repo-root .
```

### 2. 下载

```bash
modelscope download --model OneScience/RFdiffusion --local_dir ./RFdiffusion
```

### 3. 应用运行包和准备文件

下载目录本身就是标准运行包。用户可使用 `examples/input_pdbs/` 中的内置 PDB，也可以把自己的 PDB 放到任意可读路径，并通过 Hydra 参数传给 `scripts/run_inference.py`。

### 4. 运行前预检

```bash
cd ./RFdiffusion
python tools/preflight_check.py --repo-root .
```

### 5. 运行

最小无条件生成示例：

```bash
cd ./RFdiffusion
python scripts/run_inference.py \
  inference.model_directory_path=$PWD/models \
  inference.output_prefix=example_outputs/design_unconditional \
  'contigmap.contigs=[100-120]' \
  inference.num_designs=1
```

基序支架示例：

```bash
cd ./RFdiffusion
python scripts/run_inference.py \
  inference.model_directory_path=$PWD/models \
  inference.output_prefix=example_outputs/design_motifscaffolding \
  inference.input_pdb=examples/input_pdbs/1YCR.pdb \
  'contigmap.contigs=[A25-109/0 0-70/B17-29/0-70]' \
  contigmap.length=70-120 \
  inference.num_designs=1 \
  inference.ckpt_override_path=$PWD/models/Complex_base_ckpt.pt
```

### 6. 验证输出

推理成功后，`example_outputs/` 下会生成 `.pdb` 结构文件、`.trb` 元数据文件，并在 `example_outputs/traj/` 下生成扩散轨迹 PDB 文件。

## 输出说明

主要输出包括最终设计结构 PDB、包含配置和采样元数据的 TRB 文件，以及可选轨迹 PDB。无条件生成不需要输入 PDB；基序支架、PPI、酶设计和部分扩散等任务需要用户提供或选择合适的 PDB 输入与对应 checkpoint。

## 预检与诊断

| 问题 | 诊断方式 | 处理建议 |
| --- | --- | --- |
| 找不到权重 | 执行 `python tools/preflight_check.py --repo-root .` | 确认 `models/Base_ckpt.pt` 和任务需要的 checkpoint 存在，必要时重新下载模型包 |
| Manifest 解析失败 | 预检脚本会报告 YAML 错误 | 修复 `manifest.yaml` 缩进和字段 |
| repo_id 不一致 | 预检脚本会检查 `OneScience/RFdiffusion` | 统一 README、Manifest、下载命令和 relations 中的 repo_id |
| 推理依赖缺失 | 运行 `scripts/run_inference.py` 时出现 import error | 进入 OneScience 生物信息环境，安装 PyTorch、Hydra、OmegaConf 等依赖 |
| 输入 PDB 不存在 | 推理脚本报告输入路径错误 | 使用 `examples/input_pdbs/` 的内置 PDB，或提供自己的 PDB 文件绝对路径 |

## 限制与适用范围

本仓库提供 RFdiffusion 推理运行包和 checkpoint，不提供全量训练数据。复杂设计任务需要用户根据目标蛋白、contig、hotspot、对称性和 checkpoint 选择合适参数；生成结果应结合结构评估、实验约束和下游筛选继续验证。

## 引用与许可证

RFdiffusion 原始方法请引用对应论文和上游项目。本资源遵循 OneScience AI4S ModelScope 大模型运行标准。权重、代码和示例使用时请遵守原始资源许可证、OneScience 项目规则和 ModelScope 平台规则。
