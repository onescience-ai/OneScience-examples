# resnet_weather_model

## 模型介绍

resnet_weather_model是一个基于ResNet架构的天气图片单标签分类模型。模型可识别露水、雾霾、霜、雨凇、冰雹、闪电、降雨、彩虹、雾凇、沙尘暴和降雪等12类天气现象，适用于天气图片的自动识别与分类。

## 模型地址

- Hugging Face：https://huggingface.co/sallyanndelucia/resnet_weather_model

## 运行环境

- Python 3.10.12
- PyTorch 2.4.1
- Transformers 4.56.2
- 加速卡：K100AI
- 镜像：flagos_earth_onecode:v1.0.0

## 文件说明

- `config.json`：模型结构和分类标签配置
- `preprocessor_config.json`：图片预处理配置
- `snow_test.jpg`：雪景测试图片
- `download.sh`：模型权重下载脚本
- `test.ipynb`：Notebook测试代码
- `test.py`：Python测试脚本

模型权重`pytorch_model.bin`不提交到Gitee，由`download.sh`从Hugging Face下载。

## 下载模型权重

```bash
chmod +x download.sh
./download.sh
```

脚本会下载`pytorch_model.bin`并进行SHA256完整性校验。

## 运行测试

```bash
python test.py
```

也可以在JupyterLab中打开`test.ipynb`，按顺序运行全部代码单元。

## 测试结果

使用`snow_test.jpg`雪景图片进行测试：

- Top-1类别：`snow`
- Top-1概率：`0.253592`
- 输出张量形状：`(1, 12)`
- 模型加载时间：`2.2856秒`
- 推理时间：`0.014196秒`
- 测试状态：`SUCCESS`

模型可以在K100AI环境中正常运行。
