# Hurricane Image Classification Model

基于 ViT-Base/16 的飓风卫星图像分类模型。

## 模型信息

- **基础模型**: ViT-Base/16 (torchvision)
- **预训练权重**: ImageNet-1K
- **任务类型**: 图像二分类
- **数据集**: jonathan-roberts1/Satellite-Images-of-Hurricane-Damage

## 类别

- `flooded or damaged buildings` - 洪水/受损建筑
- `undamaged buildings` - 未受损建筑

## 训练配置

| 参数                 | 值                                     |
| ------------------ | ------------------------------------- |
| learning\_rate     | 2e-4                                  |
| train\_batch\_size | 16                                    |
| eval\_batch\_size  | 8                                     |
| num\_epochs        | 4                                     |
| seed               | 42                                    |
| optimizer          | AdamW (betas=(0.9, 0.999), eps=1e-08) |
| lr\_scheduler      | Linear                                |
| 预训练权重          | ImageNet-1K                    |

## 文件说明
	推理notebook
| 文件                             | 说明            |
| ------------------------------ | ------------- |
| `best_model.pth`               | 训练最佳权重 |
| `hurricane_model_complete.pth` | 完整模型   |
| `config.json`                  | 模型架构配置    |
| `preprocessor_config.json`     | 图像预处理配置   |
| `model.safetensors`            | 官方预训练权重    |
| `train_hurricane.py`           | 训练脚本          |
| `test.py`                      | 测试脚本          |
| `requirements.txt`             | Python依赖      |

## 使用方法

### 训练
```bash
python train_hurricane.py
```

### 测试
```bash
python test.py
```
## 训练结果

| Epoch | Loss | Val Acc |
|-------|------|---------|
| 1 | 0.0209 | 0.9790 |
| 2 | 0.0209 | 0.9790 |
| 3 | 0.0209 | 0.9790 |

- 最佳验证准确率: 0.9790
- 推理测试: 通过（随机输入输出形状 [1, 2]，预测类别 0）

## 数据集

- 名称: jonathan-roberts1/Satellite-Images-of-Hurricane-Damage
- 训练集: 8,000 张
- 验证集: 1,000 张（20% 的 50%）
- 测试集: 1,000 张（20% 的 50%）


## 硬件环境

- CPU: Hygon C86 7185 32-core Processor
- GPU: 无
- 内存: 32GB+

  
## 环境信息

- PyTorch: 2.4.1+
- torchvision: 0.15.0+
- datasets: 3.2.0+