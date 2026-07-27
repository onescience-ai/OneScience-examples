# RainMap Nowcasting 降水临近预报模型

## 模型简介

本项目基于 Hugging Face 上的 [TechieMoon/rainmap-nowcasting](https://huggingface.co/TechieMoon/rainmap-nowcasting) 模型完成模型加载与推理测试。
该模型为轻量级 UNet 架构，用于降水临近预报任务。输入 6 帧历史雷达回波图像，输出未来 6 帧降水预测。

### 模型特点

| 项目 | 参数 |
|------|------|
| 架构 | RainMapUNet |
| 输入帧数 | 6 |
| 输出帧数 | 6 |
| 图像尺寸 | 64 × 64 |
| 基础通道数 | 8 |
| 权重格式 | SafeTensors |

### 使用场景

- 降水临近预报演示
- 小型 UNet 架构参考
- 雷达回波序列预测实验

---

## 环境要求

### 硬件环境

| 项目 | 配置 |
|------|------|
| CPU | Hygon C86 7185 32-core Processor |
| 内存 | 32GB+ |

### 软件环境

| 软件 | 版本 |
|------|------|
| Python | 3.10.12 |
| PyTorch | 2.4.1 |
| safetensors | 最新版 |
| 基础镜像 | flagos_earth_onecode:v1.0.0 |

### 安装依赖

```bash
pip install torch safetensors
```

---

## 快速开始

### 1. 下载权重

```bash
bash download.sh
```

### 2. 运行测试

```bash
python test.py
```

### 3. 预期输出

```
==================================================
RainMap Nowcasting 模型测试
==================================================

正在加载模型...
   缺失键 (28个): ['enc1.net.1.running_mean', 'enc1.net.1.running_var', 'enc1.net.4.running_mean']
模型加载成功！
   类型: RainMapUNet
   输入帧: 6
   输出帧: 6
   图像尺寸: [64, 64]
   基础通道: 8

--------------------------------------------------
随机输入测试通过
   输入形状:  torch.Size([1, 6, 64, 64])
   输出形状:  torch.Size([1, 6, 64, 64])
   输出范围:  [-0.3494, 0.1623]
   输出均值:  -0.0719
   推理耗时:  0.4797s

==================================================
测试完成！
==================================================
```

---

## 模型架构

本模型基于标准 UNet 编码器-解码器结构：

- 编码器：3 层下采样（Conv + BatchNorm + ReLU + MaxPool）
  - 6 → 8 → 16 → 32
- 瓶颈：ConvBlock（32 → 64）
- 解码器：3 层上采样（ConvTranspose + Skip Connection + ConvBlock）
  - 64 → 32 → 16 → 8
- 输出层：1×1 卷积（8 → 6）

