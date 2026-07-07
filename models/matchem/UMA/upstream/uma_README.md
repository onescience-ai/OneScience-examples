# UMA 训练与推理 Demo

本目录提供 UMA 模型在 DCU 平台上的微调和推理示例。训练入口采用配置驱动方式：客户只需要修改 YAML 配置，不需要改 shell 启动脚本。

## 快速上手

### 1. 准备环境

首次使用前安装 MatChem 基础环境：

```bash
cd /path/to/onescience/examples/matchem
bash matchem_install.sh
```

后续每次使用：

```bash
cd /path/to/onescience/examples/matchem
source matchem_env.sh
```

确认预训练模型和数据集环境变量可用：

```bash
echo $ONESCIENCE_MODELS_DIR
echo $ONESCIENCE_DATASETS_DIR
```

默认微调配置会读取：

```text
$ONESCIENCE_MODELS_DIR/UMA/checkpoint/uma-s-1p1_converted.pt
$ONESCIENCE_DATASETS_DIR/matchem/oc20/uma_oc20_finetune/train
$ONESCIENCE_DATASETS_DIR/matchem/oc20/uma_oc20_finetune/val
```

### 2. 选择配置

`demo/configs/` 目录下提供了多个预定义训练配置：

| 配置文件 | 数据集 | GPU 数 | 说明 |
| --- | --- | --- | --- |
| `oc20_ef_4dcu.yaml` | OC20 | 4 | 单节点 4 卡，energy + forces |
| `oc20_ef_8dcu.yaml` | OC20 | 8 | 单节点 8 卡，energy + forces |
| `oc20_ef_16dcu.yaml` | OC20 | 2x8 | 双节点 16 卡，energy + forces |

### 3. 运行训练

```bash
cd /path/to/onescience/examples/matchem/uma/demo

# 直接运行，适合交互式节点或已分配资源的场景
bash run.sh --config configs/oc20_ef_4dcu.yaml

# 提交 SLURM 作业
bash run.sh --config configs/oc20_ef_4dcu.yaml --submit

# 预览命令、环境变量和 hydra 配置，不实际训练
bash run.sh --config configs/oc20_ef_4dcu.yaml --dry-run
```

多节点配置必须通过 SLURM 提交：

```bash
bash run.sh --config configs/oc20_ef_16dcu.yaml --submit
```

## 查看输出

`run.sh` 会自动生成输出目录：

```text
demo/outputs/{实验名}_{时间戳}/
```

主要文件包括：

- `config.yaml`：本次训练使用的原始配置快照
- `hydra_config.yaml`：传给 `train.py` 的最终 hydra 配置
- `submit.sh`：`--submit` 模式下生成的 SLURM 脚本
- `train_merged.out`：直接运行模式下的合并训练日志
- `uma_finetune_runs/`：训练运行目录，包含日志、结果和 checkpoint

查看日志：

```bash
tail -f outputs/{实验名}_{时间戳}/train_merged.out
```

SLURM 模式下查看 `outputs/{实验名}_{时间戳}/submit.sh` 中指定的标准输出文件，或使用 `squeue` / `sacct` 跟踪作业状态。

## 创建自定义微调实验

### 1. 准备 ASE 数据

如果已有 ASE 可读取的结构文件，并且每个 `Atoms` 对象带有能量、力等 calculator 结果，可以用脚本转换为 UMA 微调数据：

```bash
cd /path/to/onescience/examples/matchem/uma

python scripts/create_uma_finetune_dataset.py \
  --train-dir /path/to/raw_train \
  --val-dir /path/to/raw_val \
  --uma-task oc20 \
  --regression-tasks ef \
  --output-dir /path/to/uma_finetune_data \
  --num-workers 8
```

`--regression-tasks` 可选：

| 取值 | 训练目标 |
| --- | --- |
| `e` | energy |
| `ef` | energy + forces |
| `efs` | energy + forces + stress |

脚本会生成 `train/`、`val/` 以及对应的数据 YAML，并计算 `elem_refs` 和 `normalizer_rmsd`。

### 2. 复制并修改配置

```bash
cd /path/to/onescience/examples/matchem/uma/demo
cp configs/oc20_ef_4dcu.yaml configs/my_finetune.yaml
```

重点修改 `configs/my_finetune.yaml`：

| 字段 | 说明 |
| --- | --- |
| `name` | 实验名，会影响输出目录名称 |
| `launch.num_gpus` | 单节点 GPU 数 |
| `launch.num_nodes` | 节点数 |
| `env.conda_env` | conda 环境名 |
| `slurm.partition` | SLURM 分区 |
| `data.dataset_name` | UMA 任务名，例如 `oc20`、`omat`、`omol` |
| `data.elem_refs` | 元素参考能，来自数据转换脚本输出 |
| `data.normalizer_rmsd` | 归一化常数，来自数据转换脚本输出 |
| `data.train_dataset.splits.train.src` | 训练集 ASE-lmdb 路径 |
| `data.val_dataset.splits.val.src` | 验证集 ASE-lmdb 路径 |
| `runner.train_eval_unit.model.checkpoint_location` | 可微调 checkpoint 路径 |
| `epochs` / `steps` | 训练轮数或步数，二者只能一个非空 |
| `batch_size` | 每卡 batch size |

运行：

```bash
bash run.sh --config configs/my_finetune.yaml --dry-run
bash run.sh --config configs/my_finetune.yaml --submit
```

## 推理示例

`inference/` 目录提供 UMA + ASE 的推理示例。运行前请把脚本里的 checkpoint 路径改成客户环境中的实际文件。

```bash
cd /path/to/onescience/examples/matchem/uma/inference

# 无机晶体弛豫
python relax_inorganic_crystal.py

# 吸附体系弛豫与计时
python relax_adsorbate_on_slab.py

# 分子 Langevin MD
python run_molecular_md.py

# ASE-lmdb 批量推理
python batch_inference_with_dataloader.py
```

不同任务需要匹配 UMA 的 `task_name`：

| 场景 | 推荐任务 |
| --- | --- |
| 吸附/催化表面 | `oc20` |
| 无机晶体 | `omat` |
| 分子体系 | `omol` |

## 目录结构

```text
uma/
  README.md
  train.py
  configs/
    uma_sm_finetune_template.yaml
    data/
  demo/
    run.sh
    _parse_config.py
    configs/
    templates/
  inference/
  models/
  scripts/
```

## 常见问题

1. 预检提示 checkpoint 或数据路径不存在：检查 `ONESCIENCE_MODELS_DIR`、`ONESCIENCE_DATASETS_DIR`，或修改 YAML 中的绝对路径。
2. `epochs` 和 `steps` 同时设置会报错：两者只能保留一个，另一个设为 `null`。
3. 多卡训练没有拉起全部进程：检查 `launch.num_gpus`、SLURM `--gres=dcu:N` 和当前节点实际卡数是否一致。
4. 推理脚本找不到 checkpoint：示例中的 `../checkpoint/uma-s-1p1.pt` 是占位路径，需要替换为真实模型路径。
