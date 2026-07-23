---
language:
- en
license: apache-2.0
datasets:
- extreme-weather-impacts/extremeweather-types
tags:
- text-classification
- multi-label-classification
- extreme-weather
- environmental-risk
- roberta
---

# EnvironmentalBERT-extremeweather

## 模型简介

EnvironmentalBERT-extremeweather 是一个面向极端天气影响识别的英文多标签文本分类模型。模型以 EnvironmentalBERT-base 为基础，在约 4,000 条极端天气相关文本上进行微调，可判断输入文本是否涉及风暴、洪水、热浪、干旱、野火和寒潮等事件，也可以识别不包含相关事件的普通文本。

本仓库面向 OneScience 环境整理，提供模型配置、Tokenizer、权重下载脚本以及可直接运行的测试代码。测试脚本使用内置文本样本完成一次本地推理，不需要额外下载测试数据集。

## 模型信息

| 项目 | 内容 |
| --- | --- |
| 任务类型 | 英文文本多标签分类 |
| 模型结构 | `RobertaForSequenceClassification` |
| 基础模型 | `ESGBERT/EnvironmentalBERT-base` |
| 隐藏层数量 | 6 |
| 隐藏层维度 | 768 |
| 注意力头数量 | 12 |
| 最大输入长度 | 512 tokens |
| 输出类别数量 | 7 |
| 权重格式 | Safetensors |

模型输出标签如下：

| 编号 | 标签 | 含义 |
| ---: | --- | --- |
| 0 | None | 未涉及极端天气事件 |
| 1 | Storm | 风暴 |
| 2 | Flood | 洪水 |
| 3 | Heatwave | 热浪 |
| 4 | Drought | 干旱 |
| 5 | Wildfire | 野火 |
| 6 | Coldwave | 寒潮 |

## 文件结构

```text
environmentalBERT-extremeweather/
├── docs/
│   └── 测试报告.docx
├── .gitignore
├── config.json
├── download.sh
├── merges.txt
├── model_file_audit.py
├── README.md
├── special_tokens_map.json
├── test.ipynb
├── test.py
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

模型权重 `model.safetensors` 体积较大，不存放在 Gitee 仓库中。运行 `download.sh` 后，权重会下载到项目根目录；该文件已由 `.gitignore` 排除。

## 运行环境

本模型已在以下环境中完成运行验证：

| 项目 | 版本或型号 |
| --- | --- |
| 容器镜像 | `flagos_earth_onecode:v1.0.0` |
| Python | 3.10 |
| PyTorch | 2.4.1 |
| Transformers | 4.56.2 |
| 加速设备 | K100_AI |
| 运行设备 | CUDA/DCU |

主要依赖为 `torch`、`transformers` 和 `safetensors`。OneScience 镜像已经提供与加速设备匹配的 PyTorch 环境，不建议直接使用普通 PyPI 版本覆盖平台内置的 PyTorch。

## 快速开始

### 1. 进入模型目录

```bash
cd environmentalBERT-extremeweather
```

### 2. 下载模型权重

```bash
chmod +x download.sh
bash download.sh
```

下载完成后，项目根目录中应出现：

```text
model.safetensors
```

可以使用以下命令检查文件：

```bash
ls -lh model.safetensors
```

### 3. 运行测试

```bash
python test.py
```

也可以在 JupyterLab 中打开 `test.ipynb`，按照单元格顺序执行。

## 输入与输出

### 输入

输入为一条或多条英文文本。测试脚本内置了风暴、干旱与普通公司活动三类样例，并通过 Tokenizer 将文本转换为 `input_ids` 和 `attention_mask`。

### 输出

模型对每条文本输出 7 个标签的概率。由于该模型属于多标签分类，同一段文本可以同时获得多个较高概率的天气事件标签，例如一段文本可以同时涉及干旱和野火。

## 测试结果

在 K100_AI 加速设备上运行 `python test.py`，模型能够成功加载并完成三条文本的推理：

| 测试文本摘要 | 主要预测结果 | 概率 |
| --- | --- | ---: |
| Hurricanes play a significant role... | Storm | 0.960317 |
| Droughts increase the risk of severe wildfires... | Drought | 0.910172 |
| The company opened a new office... | None | 0.990351 |

本次运行记录：

- 模型加载时间：1.6706 秒
- 批量输入数量：3
- 输出张量形状：`(3, 7)`
- 推理时间：9.635308 秒
- 测试状态：`SUCCESS`

以上结果仅用于验证模型文件、运行环境和推理流程是否连通，不代表模型在正式测试集上的准确率或性能上限。

## 数据集说明

原模型使用 `extreme-weather-impacts/extremeweather-types` 数据集进行训练或微调。当前仓库的 `test.py` 使用脚本内置的少量文本进行功能验证，因此运行测试时不需要下载完整数据集。

- 数据集地址：<https://huggingface.co/datasets/extreme-weather-impacts/extremeweather-types>
- 原始模型地址：<https://huggingface.co/extreme-weather-impacts/environmentalBERT-extremeweather>

## 引用

如果在研究中使用该模型，请引用原模型对应工作：

```bibtex
@article{Schimanski25extremeweatherimpacts,
  title={What Firms Actually Lose (and Gain) from Extreme Weather Event Impacts},
  author={Tobias Schimanski and Glen Gostlow and Malte Toetzke and Markus Leippold},
  year={2025},
  journal={Available at SSRN: https://ssrn.com/abstract=6035794}
}
```

## 许可证

原模型页面标注为 Apache License 2.0。使用模型、数据集及相关代码时，请同时遵守原始模型、基础模型和数据集的许可条款。
