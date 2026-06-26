# DP 训练 Demo

本目录提供 DeepMD-kit 在 DCU 平台上的训练示例，包含 PyTorch 后端、TensorFlow 后端以及单卡/多卡 SLURM 提交脚本。推荐客户优先使用 PyTorch 后端示例。

## 快速上手

### 1. 准备环境

首次使用前先安装 MatChem 基础环境，并按需安装 DeepMD-kit：

```bash
cd /path/to/onescience/examples/matchem
bash matchem_install.sh

cd dp
bash dp_install.sh
```

后续每次使用只需加载统一环境：

```bash
cd /path/to/onescience/examples/matchem
source matchem_env.sh
dp -h
```

`dp_install.sh` 默认安装 PyTorch + TensorFlow 双后端，并安装 DeepMD C++ 接口预编译包。若生产环境不能访问外网，建议提前上传 DeepMD-kit 源码，并通过 `DEEPMD_SRC_DIR=/path/to/deepmd-kit_dcu bash dp_install.sh` 指定源码路径。

### 2. 选择示例

| 示例目录 | 后端 | GPU 数 | 说明 |
| --- | --- | --- | --- |
| `demo/water_se_e2_a_pt` | PyTorch | 1 / 4 / 8 | Water 数据集，`se_e2_a` 描述符，推荐入门 |
| `demo/water_se_atten_pt` | PyTorch | 1 / 4 / 8 | Water 数据集，`dpa1` attention 描述符 |
| `demo/water_se_e2_a_tf` | TensorFlow | 1 | Water 数据集，`se_e2_a` 描述符 |

### 3. 提交训练

以 PyTorch 后端 `se_e2_a` 示例为例：

```bash
cd /path/to/onescience/examples/matchem/dp/demo/water_se_e2_a_pt

# 单卡训练
sbatch submit_1card.sh

# 4 卡训练
sbatch submit_4card.sh

# 8 卡训练
sbatch submit_8card.sh
```

也可以在已分配的交互式节点上直接运行：

```bash
# 单卡
dp --pt train input_torch.json

# 4 卡
torchrun --nproc_per_node=4 -m deepmd --pt train input_torch.json

# 8 卡
torchrun --nproc_per_node=8 -m deepmd --pt train input_torch.json
```

TensorFlow 后端示例：

```bash
cd /path/to/onescience/examples/matchem/dp/demo/water_se_e2_a_tf
sbatch submit.sh

# 或交互式运行
dp --tf train input_tf.json
```

## 查看输出

训练输出保存在当前算例目录下，常见文件包括：

- `slurm_*.out` / `slurm_*.err`：SLURM 标准输出与错误日志
- `lcurve.out`：训练损失曲线
- `*.hdf5`：数据统计文件
- `model.ckpt*` 或 DeepMD 默认 checkpoint 文件：训练检查点

训练是否正常启动，优先查看：

```bash
tail -f slurm_*.out
tail -f lcurve.out
```

## 修改为自有数据

复制一个最接近的示例目录：

```bash
cd /path/to/onescience/examples/matchem/dp/demo
cp -r water_se_e2_a_pt my_water_pt
cd my_water_pt
```

然后修改 `input_torch.json` 中的关键字段：

| 字段 | 说明 |
| --- | --- |
| `model.type_map` | 元素类型顺序，例如 `["O", "H"]` |
| `model.descriptor.type` | 描述符类型，例如 `se_e2_a`、`dpa1` |
| `model.descriptor.rcut` | 截断半径 |
| `training.training_data.systems` | 训练集 DeepMD 数据目录 |
| `training.validation_data.systems` | 验证集 DeepMD 数据目录 |
| `training.numb_steps` | 训练步数 |
| `training.training_data.batch_size` | 训练 batch size |
| `training.disp_freq` | `lcurve.out` 输出频率 |
| `training.save_freq` | checkpoint 保存频率 |

修改后运行：

```bash
sbatch submit_1card.sh
```

多卡训练时，只需要使用对应的 `submit_4card.sh` 或 `submit_8card.sh`，训练 JSON 通常不用额外改分布式参数。

## 目录结构

```text
dp/
  README.md
  dp_install.sh
  demo/
    water_se_e2_a_pt/
      input_torch.json
      submit_1card.sh
      submit_4card.sh
      submit_8card.sh
    water_se_atten_pt/
      input_torch.json
      submit_1card.sh
      submit_4card.sh
      submit_8card.sh
    water_se_e2_a_tf/
      input_tf.json
      submit.sh
```

## 常见问题

1. `dp: command not found`：确认已经执行 `source ../matchem_env.sh`，并且 `dp_install.sh` 安装成功。
2. 数据路径不存在：检查 `input_torch.json` 或 `input_tf.json` 中的 `systems` 路径，客户环境需要替换为实际数据目录。
3. 多卡训练启动失败：确认申请的 `--gres=dcu:N` 与 `torchrun --nproc_per_node=N` 一致。
4. TensorFlow 后端训练异常：DCU 平台上推荐优先使用 PyTorch 后端；TensorFlow 后端主要用于兼容已有模型或 LAMMPS 推理链路。
