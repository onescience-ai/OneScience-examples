# GeoDiffusion - nuImages 天气/时间条件图像生成

## 模型简介
GeoDiffusion 是基于 Stable Diffusion 的目标检测数据生成模型，在 nuImages 数据集上微调，支持以一天中的时间和天气为条件生成 512x512 图像。

## 环境要求
- Python 3.10+
- PyTorch 2.0+ (DCU 适配版本)
- 参考 requirements.txt

## 使用方法
1. 运行 `python download_weights.py` 下载模型权重
2. 运行 `python run.py --prompt "your prompt"` 生成图像