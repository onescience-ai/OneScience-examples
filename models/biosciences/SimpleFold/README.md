
<p align="center">
  <strong>
    <span style="font-size: 30px;">SimpleFold</span>
  </strong>
</p>

# 模型介绍

SimpleFold 是 Apple 发布的生成式蛋白质折叠模型，用通用 Transformer 层和 flow matching 目标从蛋白质 FASTA 序列预测三维结构。当前整理包面向 ModelScope 下载、本地快速验证和 OneCode 自动化运行场景，输出结构文件支持 mmCIF/PDB。

# 仓库说明

本仓库是 OneScience 整理的 SimpleFold 最小可运行模型仓库。运行脚本默认只读取本地 `config/`、`modules/` 和 `weight/`.
当前支持能力：

- FASTA 单蛋白序列结构推理。
- 可选 pLDDT 置信度输出。
- PyTorch 后端推理；保留 MLX 后端代码，需用户自行安装 MLX 环境。
- 本地训练入口 `scripts/train.py`。
- FSDP 训练入口 `scripts/train_fsdp.py`。
- 训练数据准备入口 `scripts/process_data.py`、`scripts/tokenize_data.py`。
- 评估/预测入口 `scripts/evaluate.py`。

当前不支持能力：

- 不支持运行时自动访问远端补齐缺失文件；请保持模型包下载完整。
- 不支持蛋白-核酸/蛋白-配体复合物预测。
- 不内置真实训练数据。
- 没有独立微调脚本；微调通过 `scripts/train.py load_ckpt_path=...` 和 Hydra 配置覆盖完成。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 蛋白质结构预测 | 输入 FASTA，输出 mmCIF/PDB 结构。 |
| 本地离线推理整理 | 示例输入和权重已放入 `examples/`、`weight/`，脚本只使用本仓库文件。 |
| 训练/微调接口验证 | 用户按 `config/data/*.yaml` 准备 tokenized 数据后训练。 |
| ModelScope 标准包 | 使用 `config/ models/ scripts/ weight/` 布局。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | ModelScope 元信息 | PyTorch / 蛋白结构预测 |
| `config/` | Hydra 配置 | 由原 SimpleFold `configs/` 整理而来 |
| `models/` | SimpleFold 所需源码依赖 | 包含 simplefold、ESM、OpenFold 工具子集 |
| `scripts/run_inference.py` | 推荐推理入口 | 默认读取 `weight/` 和 `examples/minimal.fasta` |
| `scripts/inference.py` | 推理核心脚本 | 已改成本地权重模式 |
| `scripts/train.py` | 训练/微调入口 | 通过 Hydra 参数覆盖控制 |
| `scripts/train_fsdp.py` | FSDP 训练入口 | 多卡训练使用 |
| `scripts/evaluate.py` | 预测/评估入口 | 加载 checkpoint 后执行 predict |
| `scripts/process_data.py` | mmCIF 数据处理 | 训练前数据准备 |
| `scripts/tokenize_data.py` | tokenized 数据生成 | 训练前数据准备 |
| `scripts/preflight.py` | 包完整性预检 | `--strict-weights` 可强校验真实权重 |
| `weight/` | 权重和辅助文件目录 | 包含 SimpleFold、pLDDT、CCD、Boltz 辅助权重 |
| `weight/esm_models/` | ESM-2 权重目录 | 包含 ESM2 3B `.pt` 和 contact regression 权重 |
| `examples/minimal.fasta` | 最小 FASTA 示例 | 可直接替换为用户输入 |

# 权重放置

当前 ModelScope 包内已包含默认 FASTA 示例、SimpleFold 权重、pLDDT 权重、CCD 辅助文件、Boltz 辅助权重和 ESM-2 3B 本地权重，下载完整模型包后可以直接使用。

当前 `weight/` 中应包含以下真实权重和辅助文件，文件名需保持不变：

```text
weight/simplefold_100M.ckpt
weight/simplefold_360M.ckpt
weight/simplefold_700M.ckpt
weight/simplefold_1.1B.ckpt
weight/simplefold_1.6B.ckpt
weight/simplefold_3B.ckpt
weight/plddt.ckpt
weight/plddt_module_1.6B.ckpt
weight/ccd.pkl
weight/boltz1_conf.ckpt
```

推理还需要 ESM-2 3B 本地权重：

```text
weight/esm_models/esm2_t36_3B_UR50D.pt
weight/esm_models/esm2_t36_3B_UR50D-contact-regression.pt
```

也可以通过环境变量指定 ESM 主权重：

```bash
export SIMPLEFOLD_ESM2_MODEL_PATH=/path/to/esm2_t36_3B_UR50D.pt
```

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用GPU或DCU运行。
- CPU可以用于连通性验证，但速度较慢。
- DCU用户需要预先安装DTK，建议使用DTK 25.04.2以上版本或与当前集群匹配的OneScience推荐版本。

**软件要求**

想了解更多适配内容请联系 liubiao@sugon.com

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

### 1. 安装onescience库

```bash
git clone https://gitee.com/onescience-ai/onescience
cd onescience
bash install.sh bio
```

### 2. 下载模型包及权重&示例

```bash
modelscope download --model OneScience/SimpleFold --local_dir ./SimpleFold
cd SimpleFold
bash download_assets.sh
```

### 3. 运行推理

```bash
python scripts/run_inference.py \
  --simplefold_model simplefold_100M \
  --fasta_path examples/minimal.fasta \
  --output_dir outputs/minimal_inference \
  --num_steps 10 \
  --tau 0.01 \
  --nsample_per_protein 1 \
  --backend torch
```

输出目录：

```text
outputs/minimal_inference/predictions_simplefold_100M/
```

### 4. 训练

训练前准备：

```text
datasets/
datasets/tokenized/
datasets/manifest.json
```

数据处理：

```bash
python scripts/process_data.py --data_dir /path/to/mmcif --out_dir datasets --num-processes 8
python scripts/tokenize_data.py --target_dir datasets --token_dir datasets/tokenized
```

`process_data.py` 默认使用包内 `weight/ccd.pkl`，无需额外下载 CCD 或启动 Redis；如需兼容旧 Redis CCD 流程，可显式传入 `--use-redis`。

训练：

```bash
python scripts/train.py
```

FSDP 训练：

```bash
python scripts/train_fsdp.py experiment=train_fsdp
```

微调/续训示例：

```bash
python scripts/train.py load_ckpt_path=weight/simplefold_100M.ckpt
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

SimpleFold 原始实现声明为 MIT License。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。科研使用请引用 SimpleFold 原始论文和 OneScience 相关项目信息。
