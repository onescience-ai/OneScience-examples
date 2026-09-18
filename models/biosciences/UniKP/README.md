<p align="center">
  <strong><span style="font-size: 30px;">UniKP</span></strong>
</p>

# 模型介绍

UniKP 是一个基于预训练语言模型的统一酶动力学参数预测框架，可根据蛋白质序列和底物结构预测酶的周转数 $k_{cat}$、米氏常数 $K_m$ 以及催化效率 $k_{cat}/K_m$。UniKP 使用蛋白质语言模型提取酶序列表征，并结合分子语言模型生成的底物表征完成动力学参数预测。

论文：

> **UniKP: a unified framework for the prediction of enzyme kinetic parameters**  
> https://doi.org/10.1038/s41467-023-44113-1

# 模型描述

UniKP 将酶蛋白质序列与底物 SMILES 作为两路输入。蛋白质侧使用 ProtT5-XL-UniRef50 提取序列表征，底物侧使用 SMILES Transformer 生成分子表征，两路特征拼接后输入训练好的回归模型，分别预测 $k_{cat}$、$K_m$ 和 $k_{cat}/K_m$。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 酶周转数预测 | 根据蛋白质序列和底物 SMILES 预测 $k_{cat}$ |
| 米氏常数预测 | 预测酶-底物体系的 $K_m$ |
| 催化效率预测 | 预测 $k_{cat}/K_m$ |
| 酶挖掘与筛选 | 对候选酶序列进行动力学参数预测与排序 |
| 酶定向进化 | 比较野生型和突变体候选的预测动力学参数 |
| 环境因素分析 | 使用 EF-UniKP 相关实现研究温度、pH 对 $k_{cat}$ 的影响 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- ProtT5-XL-UniRef50 体积较大，蛋白质表征提取阶段推荐使用 GPU/DCU。
- 少量短序列可使用 CPU，但速度会明显慢于加速卡。
- 批量预测或长序列任务建议降低 batch size，并根据节点显存调整。

### 安装运行环境

#### DCU 环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

pip install onescience[bio] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

#### 环境说明
- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考 `requirements.txt` 中声明的依赖版本，补充安装或调整相应依赖。

### 权重准备

UniKP 完整推理额外需要两部分模型资源：

1. ProtT5-XL-UniRef50；
2. UniKP 的 $k_{cat}$、$K_m$、$k_{cat}/K_m$ 回归模型；
3. SMILES Transformer 的词表和预训练参数。

#### 1）ProtT5-XL-UniRef50

需要单独下载 ProtT5-XL-UniRef50：

```text
https://zenodo.org/records/4644188
```

建议放置为：

```text
UniKP/
└── weight/
    └── prot_t5_xl_uniref50/
```

当前代码会优先使用：

```python
T5Tokenizer.from_pretrained("weight/prot_t5_xl_uniref50")
T5EncoderModel.from_pretrained("weight/prot_t5_xl_uniref50")
```

若 `weight/prot_t5_xl_uniref50` 不存在，代码会回退使用 `"prot_t5_xl_uniref50"`，也可按实际本地路径调整 `scripts/project_paths.py`。

#### 2）UniKP 回归模型

模型下载地址：

```text
https://huggingface.co/HanselYu/UniKP/tree/main
```

典型文件包括：

```text
UniKP for kcat.pkl
UniKP for Km.pkl
UniKP for kcat_Km.pkl
```

- 已内置在 `weight/UniKP_model`。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/UniKP --local_dir ./UniKP
cd UniKP
```

- UniKP 额外依赖 **ProtT5-XL-UniRef50** 和三类 **UniKP 回归模型权重**；请先按照“权重准备”完成相关模型准备。
- SMILES Transformer 相关代码位于 `model/`，运行前请确认 `weight/vocab.pkl` 和 `weight/trfm_12_23000.pkl` 存在。

### 快速验证

检查 ProtT5 本地加载：

```bash
python - <<'PY'
from transformers import T5Tokenizer, T5EncoderModel
path = "./weight/prot_t5_xl_uniref50"
T5Tokenizer.from_pretrained(path, do_lower_case=False)
T5EncoderModel.from_pretrained(path)
print("ProtT5 load OK")
PY
```

检查回归模型和 SMILES Transformer 资源：

```bash
ls -lh weight/UniKP_model/
ls -lh weight/vocab.pkl weight/trfm_12_23000.pkl
```
若上述检查均正常，可继续运行单样本推理脚本：

```bash
python scripts/demo_kcat.py
```

# 示例数据

UniKP 推理的核心输入包括：

```text
蛋白质氨基酸序列
+
底物 SMILES
```

示例形式：

| 输入 | 示例 |
| --- | --- |
| Protein sequence | `MSELMKLSAV...MAQR` |
| Substrate SMILES | `CC(O)O` |

对应输出可以是：

```text
kcat
Km
kcat / Km
```

# 推理示例

## 单样本 kcat 预测

在 UniKP 根目录执行：

```bash
python scripts/demo_kcat.py
```

该脚本读取示例蛋白质序列和底物 SMILES，分别通过 ProtT5-XL-UniRef50 和 SMILES Transformer 提取表征，拼接后加载 `weight/UniKP_model/UniKP for kcat.pkl` 完成 kcat 预测。

预测结果会输出到终端，并保存为：

```text
UniKP_kcat_prediction.xlsx
```

`demo_kcat.py` 默认加载 kcat 回归模型：

```python
with open("weight/UniKP_model/UniKP for kcat.pkl", "rb") as f:
    model = pickle.load(f)
```

若需要预测 **Km**，将上述模型路径替换为：

```python
with open("weight/UniKP_model/UniKP for Km.pkl", "rb") as f:
    model = pickle.load(f)
```

若需要预测 **kcat/Km**，替换为：

```python
with open("weight/UniKP_model/UniKP for kcat_Km.pkl", "rb") as f:
    model = pickle.load(f)
```

除回归模型外，蛋白质表征提取、SMILES 表征提取、特征拼接和预测流程均保持不变。模型输出位于 `log10` 空间，脚本会通过 `10 ** x` 恢复为实际动力学参数。

## 批量预测

仓库分别提供 kcat、Km 和 kcat/Km 的批量预测脚本：

```bash
python scripts/UniKP_kcat.py
python scripts/UniKP_Km.py
python scripts/UniKP_kcat_Km.py
```

三个脚本分别读取对应任务的数据文件，并加载相应的 UniKP 回归模型，用于对多条蛋白质序列和底物数据进行批量预测。

运行前需确保对应任务的数据文件位于 `conf/datasets/` 中。


# 输出说明

UniKP 最终输出为对应酶动力学参数的预测值。

| 参数 | 官方示例单位 |
| --- | --- |
| $k_{cat}$ | s⁻¹ |
| $K_m$ | mM |
| $k_{cat}/K_m$ | s⁻¹·mM⁻¹ |

UniKP 回归模型内部预测值位于 `log10` 空间，因此不能直接把 `model.predict()` 结果当作实际动力学参数。应执行：

```python
pred = model.predict(fused_vector)
pred_real = [10 ** x for x in pred]
```

官方示例将结果保存为：

```text
Kinetic_parameters_predicted_label.xlsx
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |


# 引用与许可证

- UniKP 原始论文：[UniKP: a unified framework for the prediction of enzyme kinetic parameters](https://doi.org/10.1038/s41467-023-44113-1)。
- UniKP 许可信息标注为 GNU General Public License version 3（GPL-3.0）。
- UniKP 推理依赖 ProtT5-XL-UniRef50 和 SMILES Transformer 等第三方模型与代码；使用、修改或再分发时还应遵守这些资源各自的许可证和使用条款。
- 科研使用时建议引用 UniKP 原始论文；若使用 ProtT5 或 SMILES Transformer 生成表示，也应按对应项目要求补充引用。
