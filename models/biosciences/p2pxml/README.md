<p align="center">
  <strong>
    <span style="font-size: 30px;">P2PXML</span>
  </strong>
</p>

# 模型介绍

P2PXML 是用于预测抗体–抗原结合亲和力的深度几何学习框架。模型联合利用抗体与抗原的蛋白质序列信息和 PDB 三维结构信息，输出结合亲和力 IC50 的预测值。

论文：[Deep geometric framework to predict antibody–antigen binding affinity](https://doi.org/10.1016/j.jsb.2025.108257)


# 模型描述

P2PXML 的 combined model 包含两条并行分支：

- 结构分支：根据 PDB 原子坐标构建抗体图和抗原图，使用 GCN、GAT 和图池化提取结构表示；
- 序列分支：从 PDB 提取氨基酸序列并进行 one-hot 编码，通过注意力、Transformer 和跨注意力模块提取序列与相互作用表示；
- 融合输出：综合结构分支和序列分支的回归结果，预测抗体–抗原对的结合亲和力。

模型输入为一份抗体 PDB 和一份抗原 PDB。官方演示模型支持的最大序列长度为：

- 抗体：669 个氨基酸；
- 抗原：3102 个氨基酸。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 抗体–抗原结合亲和力预测 | 根据抗体和抗原 PDB 预测 IC50。 |
| 抗体候选初步筛选 | 对多组抗体–抗原结构进行亲和力趋势评估。 |
| 模型兼容性验证 | 验证 PyTorch Geometric 模型在 SCNet DCU 环境中的推理和训练能力。 |
| P2PXML 方法复现 | 使用官方 P2PXML_Structure 数据开展训练或评估。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 模型代码支持 CPU，以及 PyTorch 支持的 CUDA、ROCm/DTK 等加速设备；
- 推荐使用 GPU 或 DCU 等加速设备进行推理和训练。CPU 可用于功能验证，但 PDB 构图和模型计算速度较慢；
- 模型及图特征使用 `float64`，实际显存和内存占用取决于抗体、抗原的原子数及训练批大小，没有统一的最低显存要求；
- 完整数据训练会生成图缓存和 checkpoint，所需磁盘空间明显高于单样本推理，建议预留足够可用空间；


### 下载模型包

安装 ModelScope 命令行工具后下载包含适配代码、权重和样例数据的模型包：

```bash
python -m pip install modelscope
modelscope download --model OneScience/p2pxml --local_dir ./p2pxml
cd p2pxml
```

以下命令均应从仓库根目录执行。

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
python -m pip install -r requirements.txt
```

如需为 Notebook 注册独立内核：

```bash
python -m ipykernel install \
  --user \
  --name evo_bio \
  --display-name "Python (onescience311)"
```


### 权重与数据准备

模型包中的主要资产如下：

| 资产 | 位置 | 用途 |
| --- | --- | --- |
| 官方 combined model 权重 | `weight/model_weights.pth` | 抗体–抗原结合亲和力推理 |
| 官方样例抗体 | `conf/data/10-1074.pdb` | 快速推理输入 |
| 官方样例抗原 | `conf/data/0013095_2_11.pdb` | 快速推理输入 |
| SCNet 推理 Notebook | `scripts/smoke/P2PXML_Demonstration_SCNet.ipynb` | 本地权重和 PDB 推理 |
| 结构训练数据 | `conf/P2PXML_Structure/` | 真实数据子集或完整训练 |


官方完整数据集可从 Zenodo 下载：

https://zenodo.org/records/11531319


当前本地 `P2PXML_structure.csv` 包含 8475 条抗体–抗原记录、655 个抗体 PDB 和 469 个抗原 PDB。

### DCU/PyG 基础检查

```bash
python - <<'PY'
import torch
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(4, 4, dtype=torch.float64, device=device)
edge_index = torch.tensor(
    [[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]],
    dtype=torch.long,
    device=device,
)

gcn = GCNConv(4, 8).to(device).double()
gat = GATConv(4, 8, heads=2).to(device).double()

print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("device:", device)
print("GCN:", gcn(x, edge_index).shape)
print("GAT:", gat(x, edge_index).shape)
PY
```


### 快速推理

在已加载完整 DTK 动态库环境的 DCU 计算节点运行：

```bash
export P2PXML_ROOT=$PWD
cd scripts/smoke

P2PXML_ROOT="$P2PXML_ROOT" \
jupyter nbconvert \
  --to notebook \
  --execute P2PXML_Demonstration_SCNet.ipynb \
  --output P2PXML_Demonstration_SCNet_output.ipynb \
  --ExecutePreprocessor.kernel_name=evo_bio \
  --ExecutePreprocessor.timeout=-1
```

执行成功后，结果保存在：

```text
scripts/smoke/P2PXML_Demonstration_SCNet_output.ipynb
```


`IProgress not found` 只影响 Notebook 进度条显示，不影响推理结果。



### 真实数据子集训练

`model/integrated_model_v1_dataset_smoke.py` 支持通过环境变量选择训练样本数、epoch、输出目录和初始化权重。50条真实结构记录、1个 epoch 的验证命令如下：

```bash
cd "$P2PXML_ROOT"

P2PXML_DATA_LIMIT=50 \
P2PXML_EPOCHS=1 \
P2PXML_RUN_DIR=./scripts/training_runs/n50 \
P2PXML_INIT_CHECKPOINT=./weight/model_weights.pth \
python model/integrated_model_v1_dataset_smoke.py
```

当前上游 Dataset 在样本构图失败时会递归换用下一条样本，因此该运行能够完成，但实际有效样本可能发生重复。该结果只能说明真实数据子集训练链路可运行，不能作为严格的训练指标或数据完整性结论。

### 完整数据集单轮训练

全量结构数据的 1 epoch 命令为：

```bash
cd "$P2PXML_ROOT"

P2PXML_DATA_LIMIT=0 \
P2PXML_EPOCHS=1 \
P2PXML_RUN_DIR=./scripts/training_runs/full_epoch1 \
P2PXML_INIT_CHECKPOINT=./weight/model_weights.pth \
python model/integrated_model_v1_dataset_smoke.py
```

正式运行前应先检查 CSV 中的每个 `Ab`、`Ag` 是否存在对应 PDB，并处理缺失或异常结构。

首次构图会对 PDB 原子进行两两距离计算，复杂度接近 O(N²)，并将图缓存到 `conf/P2PXML_Structure/graph_data/`；因此完整数据首次运行的时间和磁盘开销明显高于后续 epoch。建议保留图缓存，并从50条、500条逐级扩大到全量。



# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可

- 论文：[Deep geometric framework to predict antibody–antigen binding affinity](https://doi.org/10.1016/j.jsb.2025.108257)
- 官方实现：[Drug-Discovery-ENTC/p2pxml](https://github.com/Drug-Discovery-ENTC/p2pxml)，采用 [MIT License](https://github.com/Drug-Discovery-ENTC/p2pxml/blob/main/LICENSE)。
- 官方数据：[P2PXML Dataset](https://zenodo.org/records/11531319)。官方 Notebook 标注该数据集采用 CC BY-NC-SA 4.0，使用时应遵守其非商业和署名等条款。
- 本模型包是在官方实现基础上完成的运行适配与目录整理，不改变论文、官方代码、模型权重、数据集及其他第三方资源各自的版权声明、许可证和使用条款。


