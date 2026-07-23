# NOAA ESD 珊瑚白化 ViT 分类器测试

## 模型简介

本项目基于 Hugging Face 上的 [akridge/noaa-esd-coral-bleaching-vit-classifier-v1](https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1) 模型完成模型加载与推理测试。

模型采用 Vision Transformer（ViT）架构，基于 NOAA-PIFSC 生态系统科学部（ESD）珊瑚白化分类器数据集中的图像，训练用于分类珊瑚白化状况。该数据集包含人工标注的健康和漂白珊瑚点，从而支持海洋生态系统监测的分类任务。

### 分类类别

| 类别标签 | 含义 |
|---------|------|
| CORAL | 健康珊瑚 |
| CORAL_BL | 白化珊瑚 |

### 模型性能

| 指标 | CORAL | CORAL_BL | 宏平均 |
|------|-------|----------|--------|
| 精确率 (Precision) | 0.86 | 0.84 | 0.85 |
| 召回率 (Recall) | 0.91 | 0.75 | 0.83 |
| F1 分数 | 0.88 | 0.79 | 0.84 |
| 整体准确率 | — | — | 0.85|

### 训练配置

| 配置项 | 参数 |
|--------|------|
| 基座模型 | google/vit-base-patch16-224 |
| 数据集 | NOAA ESD Coral Bleaching Classifier Dataset |
| 训练集 / 验证集 / 测试集 | 70% / 15% / 15% |
| 训练轮数 | 100 |
| 批次大小 | 16 |
| 学习率 | 3e-4 |
| 输入图像尺寸 | 224 × 224 |

---

## 文件说明

| 文件名 | 说明 |
|--------|------|
| `model.safetensors` | 模型权重文件（SafeTensors 格式） |
| `config.json` | 模型架构配置（类别数、隐藏层维度等） |
| `preprocessor_config.json` | 图像预处理配置（尺寸、归一化参数） |
| `01_example.png` | 示例图片：健康珊瑚 |
| `02_example.png` | 示例图片：白化珊瑚 |
| `test.py` | 测试脚本 |
| `test.ipynb` | 测试 Notebook |
| `download.sh` | 下载说明 |

---

## 快速开始

### 1. 准备环境

```bash
pip install torch torchvision pillow safetensors
```

### 2. 运行推理

```bash
cd /root/private_data/wyx/coral-bleaching-vit
python test.py
```

### 3. 预期输出

```
正在加载模型...
✅ 模型加载成功！权重完全匹配
类别映射: {'0': 'CORAL', '1': 'CORAL_BL'}
01_example.png: CORAL (置信度: 0.8791)
02_example.png: CORAL_BL (置信度: 0.8092)
```

---

## 测试脚本说明

### 图像预处理流程

根据 `preprocessor_config.json` 配置，预处理步骤如下：

1. 尺寸调整：将输入图像缩放为 224 × 224
2. 转张量：将像素值从 [0, 255] 映射到 [0, 1]
3. 归一化：使用均值 `[0.5, 0.5, 0.5]` 和标准差 `[0.5, 0.5, 0.5]` 进行标准化

### 模型推理流程

1. 读取 `config.json` 获取模型配置（类别数、网络结构参数）
2. 构建 ViT-Base/16 图像分类网络
3. 从 `model.safetensors` 加载权重
4. 对输入图像执行预处理
5. 前向推理，输出各类别置信度
6. 取置信度最高的类别作为预测结果
