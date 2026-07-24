# ThunderstormBERT-de-v1

## 模型介绍

ThunderstormBERT-de-v1是一个面向德语气象及雷暴文本的分类模型。模型采用DeBERTa-v2序列分类架构，输入德语气象描述后，可输出4个类别的预测概率，适用于雷暴相关历史资料和气象文本的自动分析。

## 模型地址

- Hugging Face：https://huggingface.co/Stickmu/ThunderstormBERT-de-v1

## 运行环境

- Python 3.10.12
- PyTorch 2.4.1
- Transformers 4.56.2
- 加速卡：K100AI
- 镜像：flagos_earth_onecode:v1.0.0

## 文件说明

- `config.json`：模型结构及标签配置
- `added_tokens.json`：附加词元配置
- `special_tokens_map.json`：特殊词元映射
- `tokenizer_config.json`：分词器配置
- `download.sh`：下载模型权重及分词器大文件
- `test.ipynb`：Notebook测试代码
- `test.py`：Python测试脚本

以下大文件不提交到Gitee，由`download.sh`下载：

- `model.safetensors`
- `tokenizer.json`
- `spm.model`

## 下载模型文件

```bash
chmod +x download.sh
./download.sh
```

脚本会从Hugging Face下载文件并进行SHA256完整性校验。

## 运行测试

```bash
python test.py
```

也可以在JupyterLab中打开`test.ipynb`，按顺序运行全部代码单元。

## 测试结果

使用3条德语气象文本进行测试，模型成功输出4个分类概率：

- 输出张量形状：`(3, 4)`
- 模型加载时间：`6.6898秒`
- 3条文本推理时间：`3.684189秒`
- 测试状态：`SUCCESS`

模型可以在K100AI加速卡环境中正常运行。
