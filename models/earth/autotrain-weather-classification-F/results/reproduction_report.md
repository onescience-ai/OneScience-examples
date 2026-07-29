# 天气分类模型复现报告

## 模型

`dazzle-nu/autotrain-weather-classification-3739699407`

## 数据集

`dazzle-nu/autotrain-data-weather-classification`

## 运行环境

- Python：3.10.12
- NumPy：1.26.3
- PyTorch：2.4.1
- Transformers：4.46.3
- 设备：cuda
- 设备名称：K100_AI
- 注意力实现：eager

## 本地评估结果

- 样本数量：1378
- Accuracy：0.915094
- Macro F1：0.923754
- Weighted F1：0.914670

## 模型卡公开指标

- Accuracy：0.952000
- Macro F1：0.957000
- Weighted F1：0.952000

## 复现判断

- 模型部署与推理复现：通过
- 模型卡公开指标严格复现：未通过

## 最终结论

模型部署与推理复现成功；模型卡公开指标严格复现未通过。

本次实验复现的是公开权重加载、公开验证集推理和本地指标计算。原仓库没有公开完整 AutoTrain 训练与评估代码，因此如果本地指标与模型卡指标不一致，应如实记录差异，不能把模型部署成功等同于模型卡指标严格复现成功。
