# Jena Weather TensorNEAT

## 模型介绍

本项目复现 Hugging Face 模型：

mashaalmasha/jena-weather-tensorneat

模型基于 JAX 和 TensorNEAT，使用 RecurrentGenome 神经进化方法，
根据历史气象变量预测未来气温。

## 任务定义

- 输入：过去3小时的温度、气压和相对湿度
- 每小时包含3个气象特征
- 输入维度：3 × 3 = 9
- 输出：6小时后的温度

## 正式实验配置

- 训练样本：500
- 测试样本：200
- 种群数量：300
- 进化代数：50
- 随机种子：42
- 运行设备：CPU
- Python：3.10.12
- JAX：0.6.2

## 环境安装

    python -m pip install -r requirements.txt

建议在独立虚拟环境中安装依赖。

## 下载数据

    python download_assets.py

下载完成后，数据保存在：

    data/jena_climate_2009_2016.csv

## 运行模型

    export JAX_PLATFORMS=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONHASHSEED=42

    python jena_neat.py

## 复现结果

| 指标 | 本地结果 | 官方结果 | 差值 |
|---|---:|---:|---:|
| RMSE | 3.518 °C | 3.518 °C | 0.000 °C |
| MAE | 2.797 °C | 2.797 °C | 0.000 °C |
| 基线RMSE | 4.318 °C | 4.318 °C | 0.000 °C |
| 基线MAE | 3.243 °C | 3.243 °C | 0.000 °C |
| MAE改善率 | 13.8% | 13.8% | 0.0% |

本地结果与模型卡公开到三位小数的指标一致，严格指标复现通过。

详细结果见：

- TEST_REPORT.md
- results/full_metrics.json
- results/key_results.txt
