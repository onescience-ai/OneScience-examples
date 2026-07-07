# NEP 训练 Demo

本目录提供基于 MatPL 的 NEP 训练示例，支持单卡和多卡 DCU 训练。示例脚本已经内置 MatPL 运行时环境加载逻辑，客户通常只需要安装 MatPL、选择算例并提交 `submit*.sh`。

## 快速上手

### 1. 准备环境

首次使用前安装 MatChem 基础环境，并安装 MatPL DCU：

```bash
cd /path/to/onescience/examples/matchem
bash matchem_install.sh

cd nep
bash matpl_install.sh
```

若生产环境不能访问外网，建议提前上传 MatPL DCU 源码，并通过 `MATPL_SRC_DIR` 指定：

```bash
cd /path/to/onescience/examples/matchem/nep
MATPL_SRC_DIR=/path/to/matpl_dcu bash matpl_install.sh
```

后续每次使用：

```bash
cd /path/to/onescience/examples/matchem
source matchem_env.sh
```

### 2. 选择示例

| 示例目录 | GPU 数 | 说明 |
| --- | --- | --- |
| `demo/nep_Cu` | 1 | Cu 单元素 NEP 训练，推荐入门 |
| `demo/nep_LiSiC` | 1 | Li-Si-C 多元素 NEP 训练 |
| `demo/nep_AuAg/dcu_1` | 1 | Au-Ag 单卡训练 |
| `demo/nep_AuAg/dcu_8` | 8 | Au-Ag 8 卡训练 |
| `demo/nep_AuAg/dcu_16` | 16 | Au-Ag 16 卡训练 |
| `demo/nep_HfO2/dcu_1` | 1 | HfO2 单卡训练 |
| `demo/nep_HfO2/dcu_8` | 8 | HfO2 8 卡训练 |
| `demo/nep_HfO2/dcu_16` | 16 | HfO2 16 卡训练 |

### 3. 提交训练

单卡示例：

```bash
cd /path/to/onescience/examples/matchem/nep/demo/nep_Cu
sbatch submit.sh
```

多卡示例：

```bash
cd /path/to/onescience/examples/matchem/nep/demo/nep_AuAg/dcu_8
sbatch submit_8card.sh

cd /path/to/onescience/examples/matchem/nep/demo/nep_HfO2/dcu_16
sbatch submit_16card.sh
```

也可以在已分配的交互式节点上直接运行：

```bash
cd /path/to/onescience/examples/matchem/nep/demo/nep_Cu
MatPL train Cu_nep_train.json
```

## 查看输出

训练输出保存在当前算例目录下，常见文件包括：

- `slurm_*.out` / `slurm_*.err`：SLURM 标准输出与错误日志
- MatPL 训练日志：由 `MatPL train` 在当前目录输出
- NEP 模型文件和中间结果：由 MatPL 根据训练配置生成

查看作业输出：

```bash
tail -f slurm_*.out
```

## 修改为自有数据

复制一个最接近的示例：

```bash
cd /path/to/onescience/examples/matchem/nep/demo
cp -r nep_Cu my_nep_case
cd my_nep_case
```

修改训练 JSON，例如 `Cu_nep_train.json`：

| 字段 | 说明 |
| --- | --- |
| `model_type` | 固定为 `NEP` |
| `atom_type` | 原子序数列表，例如 Cu 为 `[29]` |
| `model.descriptor.cutoff` | 径向/角向截断半径 |
| `model.descriptor.n_max` | 径向基数量 |
| `model.descriptor.basis_size` | 基函数大小 |
| `model.descriptor.l_max` | 角向阶数 |
| `model.fitting_net.network_size` | 拟合网络结构 |
| `optimizer.epochs` | 训练轮数 |
| `optimizer.batch_size` | batch size |
| `optimizer.learning_rate` | 初始学习率 |
| `optimizer.train_energy` | 是否训练能量 |
| `optimizer.train_force` | 是否训练力 |
| `optimizer.train_virial` | 是否训练 virial |
| `format` | 数据格式，当前示例为 `pwmat/movement` |
| `train_data` | 训练数据路径 |
| `valid_data` | 验证数据路径 |

如果复制后重命名了 JSON 文件，需要同步修改 `submit.sh` 末尾的命令：

```bash
MatPL train my_nep_train.json
```

## 目录结构

```text
nep/
  README.md
  matpl_install.sh
  demo/
    nep_Cu/
      Cu_nep_train.json
      std_input.json
      submit.sh
    nep_LiSiC/
      LiSiC_nep_train.json
      std_input.json
      submit.sh
    nep_AuAg/
      dcu_1/
      dcu_8/
      dcu_16/
    nep_HfO2/
      dcu_1/
      dcu_8/
      dcu_16/
```

## 常见问题

1. `MatPL: command not found`：确认 `matpl_install.sh` 已成功执行，并且 `source ../matchem_env.sh` 后 `$MATPL_SRC_DIR/env.sh` 可用。
2. 找不到 MatPL 动态库：检查 `$MATPL_SRC_DIR/src/op/build/lib` 是否存在；示例 `submit.sh` 会把它加入 `LD_LIBRARY_PATH`。
3. 数据路径不存在：将 JSON 中的 `train_data`、`valid_data` 替换为客户环境中的实际路径。
4. 多卡脚本资源不匹配：检查 `#SBATCH --gres=dcu:N`、`#SBATCH --ntasks-per-node=N` 与 JSON 中训练规模是否一致。
