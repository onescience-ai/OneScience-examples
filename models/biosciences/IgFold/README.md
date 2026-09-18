<p align="center">
  <strong>
    <span style="font-size: 30px;">IgFold</span>
  </strong>
</p>


# 模型介绍

IgFold 是 Graylab 开源的抗体结构预测模型，可根据抗体氨基酸序列快速预测抗体三维结构。模型支持重链/轻链配对抗体、单链抗体和纳米抗体预测，并可返回逐残基预测 RMSD 以及多层抗体序列表示。

论文：[Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies](https://www.nature.com/articles/s41467-023-38063-x)

# 模型描述

IgFold 使用 AntiBERTy 提取抗体序列表示，通过图 Transformer、模板特征融合和不变点注意力模块预测抗体结构。模型提供以下主要能力：

- 根据 H/L 双链序列预测配对抗体结构；
- 根据单条重链或轻链序列预测单链抗体或纳米抗体结构；
- 输出每个残基 N、CA、C、CB 原子的预测 RMSD；
- 输出 AntiBERTy、图 Transformer 和结构模块的中间表示；
- 支持使用模板结构；
- 支持使用 PyRosetta 或 OpenMM 进行结构精修；
- 支持将预测结构转换为 Chothia 编号。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 配对抗体结构预测 | 根据重链和轻链序列生成抗体 PDB 结构。 |
| 单链抗体结构预测 | 根据单条重链或轻链序列预测单链抗体结构。 |
| 纳米抗体结构预测 | 根据纳米抗体重链序列生成 PDB 结构。 |
| 预测误差分析 | 获取逐残基预测 RMSD，并在输出 PDB 的 B-factor 列中查看对应结果。 |
| 抗体序列表示 | 获取 AntiBERTy、图 Transformer 和结构模块的嵌入表示。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 模型支持在 CPU 或 PyTorch 支持的加速设备上运行；
- 推荐使用 GPU 等加速设备进行结构预测；
- PyRosetta 精修主要使用 CPU，运行时间与输入序列长度和硬件性能有关；
- 实际可用设备和安装方式取决于用户本地的 PyTorch、驱动及运行环境。

### 下载模型包

安装 ModelScope 命令行工具后下载模型：

```bash
pip install modelscope
modelscope download --model OneScience/IgFold --local_dir ./IgFold
cd IgFold
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

请根据 `requirements.txt` 中声明的依赖版本额外安装依赖

```bash
conda activate onescience311
python -m pip install -r requirements.txt
```

模型包的 `weight/wheels` 目录还提供包含官方预训练资产的 IgFold 和 AntiBERTy wheel，可用于离线安装：

```bash
python -m pip install --no-deps \
  weight/wheels/antiberty-0.1.3-py3-none-any.whl \
  weight/wheels/igfold-0.4.0-py3-none-any.whl
```

### 权重与数据准备

基础结构推理需要 IgFold 四个官方 checkpoint 和 AntiBERTy 预训练权重。模型包已经提供推理所需资产，无需在运行时额外下载权重：

| 资产 | 位置 | 用途 |
| --- | --- | --- |
| IgFold checkpoint | `weight/IgFold/igfold_1.ckpt` | 四模型结构预测集成成员 |
| IgFold checkpoint | `weight/IgFold/igfold_2.ckpt` | 四模型结构预测集成成员 |
| IgFold checkpoint | `weight/IgFold/igfold_3.ckpt` | 四模型结构预测集成成员 |
| IgFold checkpoint | `weight/IgFold/igfold_5.ckpt` | 四模型结构预测集成成员 |
| AntiBERTy 模型和权重 | `weight/wheels/antiberty-0.1.3-py3-none-any.whl` | 生成抗体序列表示 |

`scripts/inference.py` 默认从 `weight/IgFold/` 加载四个 checkpoint；AntiBERTy 权重会随对应 wheel 安装到 Python 环境中。完成前述离线安装后，推理过程不需要联网下载模型。

基础推理只需要抗体重链/轻链氨基酸序列。未提供输入时，推理脚本会使用官方示例序列；使用自定义序列时，准备一个链标识为 `H` 和 `L` 的 FASTA 文件即可：

```text
>sample:H
EVQLVQSGPEVKKPGTSVKVSCKAS...
>sample:L
DVVMTQTPFSLPVSLGDQASISCR...
```

### 可选依赖

#### PyRosetta 精修

IgFold 支持使用 PyRosetta 对预测结构进行精修。请按照 [PyRosetta 官方说明](https://www.pyrosetta.org/downloads) 安装与当前 Python 和操作系统匹配的版本。

#### OpenMM 精修

如果不使用 PyRosetta，也可以安装 OpenMM 和 PDBFixer：

```bash
conda install -c conda-forge openmm==7.7.0 pdbfixer
```

#### Chothia 编号

如需将预测结构转换为 Chothia 编号，请安装 AbNumber：

```bash
conda install -c bioconda abnumber
```

### 快速推理

模型包提供了可直接运行的推理入口。未指定序列或 FASTA 时，脚本会使用官方 README 中的 H/L 双链示例，并将结构写入 `output/inference/antibody.pdb`：

```bash
python scripts/inference.py
```

使用自己的 FASTA：

```bash
python scripts/inference.py \
  --fasta /path/to/antibody.fasta \
  --output output/inference/my_antibody.pdb
```

FASTA 中的链标识应为 `H` 和 `L`，例如：

```text
>sample:H
EVQLVQSGPEVKKPGTSVKVSCKAS...
>sample:L
DVVMTQTPFSLPVSLGDQASISCR...
```

不启用可选参数时，快速推理不需要 SAbDab PDB 数据、PyRosetta、OpenMM 或 AbNumber。需要精修或重新编号时，可在安装相应依赖后使用 `--refine`、`--openmm` 和 `--renum`。

### 配对抗体结构预测

重链和轻链序列以字典形式传入，键分别为 `H` 和 `L`：

```python
from igfold import IgFoldRunner
from igfold.refine.pyrosetta_ref import init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK",
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb,
    sequences=sequences,
    do_refine=True,
    do_renum=True,
)
```

运行完成后，预测结构将保存到 `my_antibody.pdb`。

### 纳米抗体或单链抗体结构预测

预测纳米抗体或单条重链/轻链结构时，只需提供一条序列：

```python
from igfold import IgFoldRunner
from igfold.refine.pyrosetta_ref import init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "QVQLQESGGGLVQAGGSLTLSCAVSGLTFSNYAMGWFRQAPGKEREFVAAITWDGGNTYYTDSVKGRFTISRDNAKNTVFLQMNSLKPEDTAVYYCAAKLLGSSRYELALAGYDYWGQGTQVTVS",
}
pred_pdb = "my_nanobody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb,
    sequences=sequences,
    do_refine=True,
    do_renum=True,
)
```

### 不使用结构精修

不需要 PyRosetta 或 OpenMM 精修时，将 `do_refine` 设置为 `False`。如果也不需要 Chothia 编号，可以同时将 `do_renum` 设置为 `False`，此时无需安装对应的可选依赖：

```python
from igfold import IgFoldRunner

sequences = {
    "H": "QVQLQESGGGLVQAGGSLTLSCAVSGLTFSNYAMGWFRQAPGKEREFVAAITWDGGNTYYTDSVKGRFTISRDNAKNTVFLQMNSLKPEDTAVYYCAAKLLGSSRYELALAGYDYWGQGTQVTVS",
}
pred_pdb = "my_nanobody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb,
    sequences=sequences,
    do_refine=False,
    do_renum=False,
)
```

### 预测 RMSD

IgFold 会计算逐残基预测 RMSD，并将结果记录在输出 PDB 的 B-factor 列中。该结果也会由 `fold()` 返回：

```python
from igfold import IgFoldRunner

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK",
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
out = igfold.fold(
    pred_pdb,
    sequences=sequences,
    do_refine=False,
    do_renum=False,
)

print(out.prmsd)
# 每个残基 N、CA、C、CB 原子的预测 RMSD，形状为 [1, L, 4]
```

这里的 `prmsd` 是模型预测的误差估计，不是与实验结构对齐后计算得到的真实 RMSD。

### 抗体序列嵌入

`embed()` 方法可以输出多个层级的抗体表示：

```python
from igfold import IgFoldRunner

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK",
}

igfold = IgFoldRunner()
emb = igfold.embed(sequences=sequences)

print(emb.bert_embs.shape)       # AntiBERTy 最后一层表示：[1, L, 512]
print(emb.gt_embs.shape)         # 图 Transformer 表示：[1, L, 64]
print(emb.structure_embs.shape)  # 结构模块表示：[1, L, 64]
```

### 优先使用 OpenMM 精修

安装 OpenMM 和 PDBFixer 后，可以通过 `use_openmm=True` 优先使用 OpenMM：

```python
from igfold import IgFoldRunner

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK",
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb,
    sequences=sequences,
    do_refine=True,
    use_openmm=True,
    do_renum=True,
)
```

### 训练

IgFold 官方仓库没有提供可直接运行的完整训练入口、数据集类或训练脚本，因此本模型包不提供训练命令。`model/training` 中只包含模型计算损失时使用的工具函数，不能视为完整训练程序。

论文使用的官方训练结构数据已发布在 Zenodo：

https://doi.org/10.5281/zenodo.7820263

该数据包括实验测定的 SAbDab 抗体结构，以及基于 OAS 配对和非配对序列生成的结构。它们用于复现论文训练或开展结构评估。

如需自行复现训练，应结合论文方法、上游模型损失函数和官方训练数据另行实现 Dataset、DataLoader、优化器、训练循环及 checkpoint 管理。

### 合成抗体结构数据

IgFold 官方还发布了两组大规模预测结构：

- 104K 条 OAS 非冗余配对抗体预测结构：https://data.graylab.jhu.edu/OAS_paired.tar.gz
- 1.3M 条 Jaffe 等人收集的人类配对抗体预测结构：https://data.graylab.jhu.edu/Jaffe2022.tar.gz

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies](https://www.nature.com/articles/s41467-023-38063-x)。

- 官方实现：[Graylab/IgFold](https://github.com/Graylab/IgFold)，代码、数据和预训练模型依据 [JHU Academic Software License Agreement](https://github.com/Graylab/IgFold/blob/main/LICENSE.md) 的非商业条款提供；商业用途需通过 Johns Hopkins Technology Ventures 获取许可。

- 本模型包基于 IgFold 官方实现进行整理和发布，不代表论文作者或 Graylab 的新增官方版本。论文、官方代码、预训练模型、AntiBERTy、PyRosetta、SAbDab、OAS 及其他第三方资源分别受其原始版权声明、许可证和使用条款约束。
