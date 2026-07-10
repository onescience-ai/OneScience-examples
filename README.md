

# OneScience Examples

欢迎来到 OneScience Examples 仓库！本仓库汇集了多种前沿人工智能/机器学习模型的示例代码、训练脚本和推理流程，涵盖蛋白质结构预测、分子动力学、计算流体力学、天气预报等多个领域。

## 项目简介

OneScience Examples 是 OneScience 官方维护的模型示例仓库，旨在为研究人员和开发者提供开箱即用的 AI/ML 模型解决方案。每个子项目都包含了完整的运行环境配置、数据准备脚本、训练/推理代码以及详细的使用文档。

## 可用模型

本仓库目前支持以下模型类别：

### 🧬 生物科学

| 模型 | 描述 |
|------|------|
| [AlphaFold3](./models/AlphaFold3/README.md) | DeepMind 第三代蛋白质结构预测模型 |
| [AlphaGenome](./models/alphagenome/README.md) | DNA 序列分析与变异评分模型 |
| [ESM](./models/ESM/README.md) | ESMFold 蛋白质结构预测 |
| [Evo2](./models/evo2/README.md) | 大规模基因组基础模型 |
| [OpenFold](./models/OpenFold/README.md) | 开源蛋白质结构预测 |
| [PINNsformer](./models/PINNsformer/README.md) | 物理-informed 神经网络 |
| [Protenix](./models/protenix/README.md) | 蛋白质结构预测模型 |
| [ProteinMPNN](./models/ProteinMPNN/README.md) | 蛋白质序列设计 |
| [RFdiffusion](./models/RFdiffusion/README.md) | 蛋白质反向折叠扩散模型 |
| [SimpleFold](./models/SimpleFold/README.md) | 轻量级蛋白质结构预测 |

### 🧪 材料化学

| 模型 | 描述 |
|------|------|
| [DeepMD](./models/matchem/DeepMD/README.md) | 深度势能分子动力学 |
| [MACE](./models/matchem/MACE/README.md) | 原子间势能模型 |
| [MatRIS](./models/matchem/MatRIS/README.md) | 材料发现与结构预测 |
| [NEP](./models/matchem/NEP/README.md) | 神经网络原子势能 |
| [UMA](./models/matchem/UMA/README.md) | 统一分子架构 |

### 🌤️ 天气预报与气候

| 模型 | 描述 |
|------|------|
| [FourCastNet](./models/FourCastNet/README.md) | 图像天气预报模型 |
| [FuXi](./models/FuXi/README.md) | 气象预报模型 |
| [FengWu](./models/FengWu/README.md) | 气象预测模型 |
| [GraphCast](./models/GraphCast/README.md) | 图神经网络天气预报 |
| [Pangu-Weather](./models/Pangu_Weather/README.md) | 盘古气象大模型 |
| [XiHe](./models/XiHe/README.md) | 气象预测模型 |

### 💧 计算流体力学 (CFD)

| 模型 | 描述 |
|------|------|
| [CFDBench](./models/CFDBench/README.md) | CFD 基准数据集 |
| [DeepCFD](./models/DeepCFD/README.md) | 深度学习 CFD 模型 |
| [EagleMeshTransformer](./models/EagleMeshTransformer/README.md) | 网格图神经网络 |
| [GP_for_TO](./models/GP_for_TO/README.md) | 高斯过程优化 |
| [LagrangianMGN](./models/LagrangianMGN/README.md) | 拉格朗日图网络 |
| [MeshGraphNet](./models/MeshGraphNet/README.md) | 网格图神经网络 |

### 🎨 设计与生成

| 模型 | 描述 |
|------|------|
| [Transolver-Airfoil-Design](./models/Transolver-Airfoil-Design/README.md) | 翼型设计 |
| [Transolver-Car-Design](./models/Transolver-Car-Design/README.md) | 汽车设计 |

### 📐 偏微分方程神经网络 (PDENN)

| 模型 | 描述 |
|------|------|
| [DeepONet](./models/PDENNEval/DeepONet/README.md) | 深度算子网络 |
| [FNO](./models/PDENNEval/FNO/README.md) | 傅里叶算子网络 |
| [MPNN](./models/PDENNEval/MPNN/README.md) | 消息传递神经网络 |
| [PINN](./models/PDENNEval/PINN/README.md) | 物理-informed 神经网络 |
| [PINO](./models/PDENNEval/PINO/README.md) | 物理算子网络 |
| [UNO](./models/PDENNEval/UNO/README.md) | 统一算子网络 |
| [U-Net](./models/PDENNEval/UNet/README.md) | U-Net |
| [WAN](./models/PDENNEval/WAN/README.md) | 波自适应网络 |

## 快速开始

### 1. 环境准备

每个模型项目都有其特定的环境要求。请参考各模型目录下的 README.md 获取详细的环境安装指南。

通用依赖：
```bash
# 基础环境 (根据具体模型可能需要额外配置)
conda create -n onescience python=3.10
conda activate onescience
pip install torch torchvision
```

### 2. 下载模型和数据

大部分模型需要下载预训练权重和数据集。使用各模型目录下的下载脚本：

```bash
cd models/<MODEL_NAME>
bash download.sh
```

### 3. 运行示例

参考各模型 README.md 中的运行流程，一般流程包括：

1. **环境预检** - 验证运行环境
2. **数据准备** - 下载和解压数据
3. **运行推理** - 执行预测
4. **验证输出** - 检查结果

## 项目结构

```
onescience-examples/
├── datasets/              # 数据集相关文档
├── models/              # 模型代码
│   ├── AlphaFold3/      # 蛋白质结构预测
│   ├── FourCastNet/    # 天气预报
│   ├── DeepMD/         # 分子动力学
│   ├── GraphCast/      # 图神经网络天气
│   ├── PDENNEval/      # 偏微分方程网络
│   └── ...             # 其他模型
└── README.md           # 本文件
```

## 文档说明

每个模型项目包含以下标准文档：

- **README.md** - 项目说明、安装指南、使用教程
- **manifest.yaml** - 模型文件清单
- **conf/** - 配置文件目录
- **scripts/** - 辅助脚本目录
- **train.py / inference.py** - 训练/推理入口

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本仓库：

1. 提交 Issue 报告问题或提出新模型需求
2. Fork 本仓库
3. 创建新分支进行修改
4. 提交 Pull Request

## 许可证

本仓库中的代码遵循各模型原始项目的许可证。具体请参考各模型目录下的 LICENSE 文件或 README.md 中的许可说明。

## 联系方式

- 官方网站：https://onescience.ai
- Gitee：https://gitee.com/onescience-ai
- GitHub：https://github.com/onescience-ai

## 致谢

感谢所有开源模型作者和 OneScience 团队的贡献。