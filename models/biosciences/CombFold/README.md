<p align="center">
  <strong>
    <span style="font-size: 30px;">CombFold</span>
  </strong>
</p>


# 模型介绍

CombFold 是 dina-lab3D 开源的大型蛋白质复合物结构预测流程，可从复合物各条链的氨基酸序列出发，利用 AlphaFold-Multimer 预测多个候选子复合物，再通过组合组装算法构建完整复合物。论文报告的处理规模可达到至少 18,000 个氨基酸和 32 个亚基。

论文：[Assembly of protein complexes by combining AlphaFold and combinatorial optimization](https://www.nature.com/articles/s41592-024-02174-0)

# 模型描述

CombFold 包含以下四个阶段：

1. 根据蛋白质结构域和链组成定义子单元，生成 `subunits.json`；
2. 为所有子单元配对生成 FASTA，并使用 AlphaFold-Multimer 预测配对子复合物；
3. 可选地预测包含更多子单元的候选子复合物；
4. 从预测 PDB 中提取子单元相对变换，并使用 C++ 组合算法组装完整复合物。

CombFold 本身不是需要训练的神经网络。神经网络推理由预训练 AlphaFold-Multimer 完成，CombFold 负责结构变换提取和组合组装。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 大型蛋白质复合物预测 | 将多个 AFM 子复合物预测组合为完整复合物。 |
| 同源多聚体预测 | 根据唯一子单元及其化学计量关系组装多拷贝复合物。 |
| 异源复合物预测 | 综合不同子单元配对或分组预测获得整体结构。 |
| 已有 AFM 结果组装 | 直接输入已有 AlphaFold-Multimer PDB，无需重新运行 AFM。 |
| 交联约束组装 | 可选输入交联约束，辅助限制候选组装结构。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- CombFold C++ 组合组装阶段只需要 CPU；
- 本地生成 AlphaFold-Multimer 子复合物通常需要加速设备；

- 长序列和多模型推理的显存占用随总残基数、MSA 深度、模型数量和 recycle 数增加；
- PyTorch 不是 CombFold 或当前 ColabFold 推理链路的直接运行依赖。

### 下载模型包

安装 ModelScope 命令行工具后下载模型：

```bash
pip install modelscope
modelscope download --model OneScience/CombFold --local_dir ./CombFold
cd CombFold
```


### 安装运行环境

**OneScience DCU 基础环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
python -m pip install onescience[bio-dcu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

安装额外依赖：

```bash
python -m pip install --no-deps -r requirements.txt
```



### 编译组合组装器

CombFold 的组合组装阶段由 C++17 实现，需要以下系统级组件；这些组件不能由 `requirements.txt` 安装：

| 组件 | 作用 | 说明 |
| --- | --- | --- |
| C++17 编译器 | 编译 C++ 源码 | Linux 通常使用 `g++` |
| GNU Make | 执行 Makefile | 已验证 GNU Make 4.2.1 |
| Boost headers | 编译期头文件 | 目录中必须包含 `boost/algorithm/string.hpp` |
| Boost program_options | 链接期运行库 | Linux 下通常为 `libboost_program_options.so` |

Boost headers 大体可以跨 Linux 发行版使用，但编译后的 Boost 库与操作系统、CPU 架构、编译器及 libstdc++ ABI 相关，不能直接在 Linux、macOS、Windows 或不同 CPU 架构间复制。建议 headers 与运行库使用相同 Boost 版本。

上游 Makefile 的默认配置面向 macOS Homebrew。在 Linux 上，只有 Boost 已处于编译器默认搜索路径时，才能直接执行：

```bash
cd model/CombinatorialAssembler
make
cd ../..
```

#### 指定自定义 Boost 路径

如果 Boost 不在默认搜索路径中，设置以下两个环境变量：

```bash
export COMBFOLD_BOOST_INCLUDE="<Boost源码或include目录>"
export COMBFOLD_BOOST_LIB="<Boost库目录>"
```

`COMBFOLD_BOOST_INCLUDE` 应指向直接包含 `boost/` 子目录的位置；`COMBFOLD_BOOST_LIB` 应指向直接包含 `libboost_program_options` 的位置。编译前可以检查：

```bash
test -f "${COMBFOLD_BOOST_INCLUDE}/boost/algorithm/string.hpp" \
  && echo "Boost headers OK"

find "${COMBFOLD_BOOST_LIB}" -maxdepth 1 \
  -name 'libboost_program_options*' -print
```

编译：

```bash
cd model/CombinatorialAssembler

make -j4 \
  BOOST_INCLUDE="${COMBFOLD_BOOST_INCLUDE}" \
  BOOST_LIB="${COMBFOLD_BOOST_LIB}"

cd ../..
```




### 权重与数据准备

CombFold 组合组装器本身没有权重。本地生成子复合物需要五个官方 AlphaFold-Multimer v3 参数文件：

| 资产 | 模型包内位置 | 用途 |
| --- | --- | --- |
| `params_model_1_multimer_v3.npz` | `weight/alphafold/params/` | AFM v3 模型 1 |
| `params_model_2_multimer_v3.npz` | `weight/alphafold/params/` | AFM v3 模型 2 |
| `params_model_3_multimer_v3.npz` | `weight/alphafold/params/` | AFM v3 模型 3 |
| `params_model_4_multimer_v3.npz` | `weight/alphafold/params/` | AFM v3 模型 4 |
| `params_model_5_multimer_v3.npz` | `weight/alphafold/params/` | AFM v3 模型 5 |



```text
--data weight/alphafold
```

如果只使用已有 AFM PDB 进行组合组装，则不需要下载 AlphaFold-Multimer 权重。

### 定义子单元

输入 `subunits.json` 是以子单元名称为键的 JSON 字典。每个子单元包含：

- `name`：唯一子单元名称；
- `sequence`：氨基酸序列；
- `chain_names`：该子单元在完整复合物中的链名，列表长度同时表示化学计量；
- `start_res`：该序列在原始链中的起始残基编号。

示例：

```json
{
  "A0": {
    "name": "A0",
    "chain_names": ["A", "B"],
    "start_res": 1,
    "sequence": "MKDILEKLEERRAQARLGGGEKRLEAQHKRGKLTARERIELLLDHGSFEE"
  }
}
```

模型包提供完整示例：

```text
scripts/example/subunits.json
scripts/example/pdbs/
```

### 快速推理：使用已有 PDB 进行 CPU 组装

这是最短的 CombFold 推理路径，不运行 AlphaFold-Multimer：

```bash
python scripts/inference.py \
  --subunits scripts/example/subunits.json \
  --pdbs scripts/example/pdbs \
  --output output/example_assembly
```

输出目录必须不存在或为空。成功后结果位于：

```text
output/example_assembly/assembled_results/output_clustered_0.pdb
output/example_assembly/assembled_results/confidence.txt
```

也可以使用上游原始位置参数入口：

```bash
python scripts/run_on_pdbs.py \
  scripts/example/subunits.json \
  scripts/example/pdbs \
  output/example_assembly
```

### 生成配对 FASTA

根据 `subunits.json` 为所有唯一子单元配对生成 FASTA：

```bash
python scripts/prepare_fastas.py \
  scripts/example/subunits.json \
  --stage pairs \
  --output-fasta-folder output/pair_fastas \
  --max-af-size 1800
```

输出目录必须事先不存在。官方示例会生成：

```text
A0_A0.fasta
A0_G0.fasta
G0_G0.fasta
```

### GPU 最小配对推理

计算节点离线 smoke test 可使用 `single_sequence`、单模型和一次 recycle：

```bash
colabfold_batch \
  output/pair_fastas \
  output/colabfold_pairs \
  --data weight/alphafold \
  --model-type alphafold2_multimer_v3 \
  --model-order 1 \
  --num-models 1 \
  --num-recycle 1 \
  --num-relax 0 \
  --msa-mode single_sequence \
  --disable-unified-memory
```

`single_sequence + 1 model + 1 recycle` 只用于验证权重加载、JAX/DCU 前向和 PDB 输出，不用于评价正式预测精度。正式预测建议准备 MSA，并根据显存和耗时增加模型数与 recycle 数。



### GPU 到 CPU 的端到端推理

每个配对至少选择一个预测 PDB。使用 ColabFold 排名第一的结构：

```bash
mkdir -p output/combfold_pdbs

find output/colabfold_pairs -maxdepth 1 \
  -type f -name '*rank_001*.pdb' \
  -exec cp {} output/combfold_pdbs/ \;
```

然后运行组合组装：

```bash
python scripts/inference.py \
  --subunits scripts/example/subunits.json \
  --pdbs output/combfold_pdbs \
  --output output/end2end_assembly
```

统一入口会输出机器可读摘要，例如：

```text
COMBFOLD_INFERENCE_RESULT={"assembled_structures": 5, "format": "pdb", "status": "PASS", ...}
```

### 使用交联约束

通过 `--crosslinks` 输入交联文件：

```bash
python scripts/inference.py \
  --subunits scripts/example/example_xlinks/subunits.json \
  --pdbs scripts/example/example_xlinks/pdbs \
  --crosslinks scripts/example/example_xlinks/crosslinks.txt \
  --output output/crosslink_assembly
```

### 可选的更大子复合物预测

配对预测完成后，可以根据配对结果生成更大组合的 FASTA：

```bash
python scripts/prepare_fastas.py \
  scripts/example/subunits.json \
  --stage groups \
  --output-fasta-folder output/group_fastas \
  --max-af-size 1800 \
  --input-pairs-results output/combfold_pdbs
```





### 训练

CombFold 是基于预训练 AlphaFold-Multimer 结果进行组合组装的推理算法，本身不包含可训练模型、训练入口、优化器或训练数据流程，因此本模型包不提供训练命令。

重新训练 AlphaFold-Multimer 属于独立的上游大模型训练任务，不属于 CombFold 的组合组装流程。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[Assembly of protein complexes by combining AlphaFold and combinatorial optimization](https://www.nature.com/articles/s41592-024-02174-0)。

- 官方实现：[dina-lab3D/CombFold](https://github.com/dina-lab3D/CombFold)，代码依据仓库中的 Apache License 2.0 提供。

- AlphaFold、AlphaFold-Multimer、ColabFold、预训练参数及其他第三方组件分别受其原始版权声明、模型条款和许可证约束。
