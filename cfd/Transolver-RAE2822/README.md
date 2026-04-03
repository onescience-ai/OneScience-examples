# Transolver 模型适配 RAE2822 数据集 - AI4S 智能体生成案例

## 概述

本案例展示了如何使用 OneScience 框架的 AI4S 智能体，将 Transolver 模型从 AirfRANS 翼型数据集适配到 RAE2822 翼型数据集。所有代码均由智能体基于 oneskills 自动生成，包括数据集接口、配置文件、训练脚本和推理脚本。

## 生成说明

### 生成方式

本案例使用 OneScience 框架的 `ai4s-case-generator` skill 自动生成，智能体根据以下输入完成了完整的适配工作：

- **数据集**: RAE2822 翼型数据集
- **目标模型**: Transolver
- **参考模板**: `onescience/examples/cfd/Transolver-Airfoil-Design/`
- **输出目录**: `/root/private_data/test/`

### 生成流程

智能体按照以下步骤自动生成代码：

1. **分析数据集 README**: 理解 RAE2822 数据集的结构、字段和格式
2. **选择参考模板**: 选择最接近的 AirfRANS datapipe 作为模板
3. **生成 datapipe**: 创建 `RAE2822.py`，包含 `RAE2822Dataset` 和 `RAE2822Datapipe`
4. **生成配置**: 创建 `conf/transolver_rae2822.yaml`，适配数据集参数
5. **生成训练脚本**: 创建 `train.py`，复用 Transolver 训练流程
6. **生成推理脚本**: 创建 `inference.py`，支持模型推理和结果保存
7. **代码审查**: 自动检查未定义变量、shape 对齐、导入路径等问题

### 参考的 oneskills

智能体在生成过程中使用了以下 oneskills：

- `ai4s-case-generator`: 总控入口，协调整个生成流程
- `dataset-interface-generator`: 生成数据集接口文件
- `onescience-datapipe-code-standard`: 确保代码符合 OneScience 规范
- `datapipe-generation-patterns`: 提供数据管道生成模式
- `config-generator`: 生成模型配置文件
- `train-script-generator`: 生成训练脚本
- `inference-script-generator`: 生成推理脚本
- `generated-case-review`: 生成后代码审查

## 数据集说明

### RAE2822 翼型数据集

RAE2822 数据集包含跨音速 RAE2822 翼型的计算流体力学（CFD）模拟结果，数据来源于论文：

**"A comparative study of learning techniques for the compressible aerodynamics over a transonic RAE2822 airfoil"**  
*Computational Fluid Dynamics Journal*, 2022  
DOI: [10.1016/j.compfluid.2022.105759](https://doi.org/10.1016/j.compfluid.2022.105759)

### 数据集特征

- **翼型**: RAE2822（经典跨音速翼型基准）
- **流动状态**: 跨音速（Mach 0.0 到 0.9）
- **攻角范围**: 0° 到 9°
- **网格类型**: 结构化计算网格
- **变量**: 压力、速度分量（Vx, Vy）、坐标

### 数据文件

数据集包含以下文件：

1. **airfoil.npy**: RAE2822 翼型坐标定义
2. **db_random.npy**: 使用随机采样的数据集
   - Mach 数：0.0 到 0.9（均匀随机分布）
   - 攻角：0° 到 9°（均匀随机分布）
   - 采样方法：蒙特卡洛随机采样
3. **db_cyc.npy**: 使用 Clenshaw-Curtis 积分规则的数据集
   - Mach 数：0.0 到 0.9（结构化采样）
   - 攻角：0° 到 9°（结构化采样）
   - 采样方法：Clenshaw-Curtis 积分规则，更好的参数空间覆盖

### 数据结构

每个数据集文件（`.npy`）包含一个字典，具有以下键：

| 键 | 描述 | 形状 |
|-----|------|------|
| `Pressure` | 计算网格上的压力场 | `[n_samples, grid_x, grid_y]` |
| `Vx` | 速度场的 X 分量 | `[n_samples, grid_x, grid_y]` |
| `Vy` | 速度场的 Y 分量 | `[n_samples, grid_x, grid_y]` |
| `Xcoordinate` | 网格点的 X 坐标 | `[grid_x, grid_y]` |
| `Ycoordinate` | 网格点的 Y 坐标 | `[grid_x, grid_y]` |
| `Vinf` | 每个样本的自由流 Mach 数 | `[n_samples]` |
| `Alpha` | 每个样本的攻角（度） | `[n_samples]` |
| `idx` | 样本索引 | `[n_samples]` |

### 数据预处理

为了适配 Transolver 模型，数据经过以下预处理：

1. **输入特征构建**（7 维）:
   - `pos_x, pos_y`: 网格点坐标
   - `vinf`: 自由流 Mach 数（广播到所有点）
   - `alpha`: 攻角（弧度，广播到所有点）
   - `nx, ny`: 法向量（简化为 0，因为数据集中没有）
   - `boundary_mask`: 边界掩码（简化为 0）

2. **输出特征构建**（4 维）:
   - `pressure`: 压力场
   - `vx`: X 方向速度
   - `vy`: Y 方向速度
   - `nu`: 湍流粘度（设为 0，因为数据集中没有）

3. **归一化**:
   - 统计量在训练集上计算
   - 验证集和测试集复用训练集统计量
   - 使用 Z-score 标准化：`(x - mean) / (std + 1e-8)`

## 训练流程

### 环境准备

确保已安装以下依赖：

```bash
pip install torch torch-geometric numpy pyyaml tqdm
```

### 训练步骤

1. **进入项目目录**:

```bash
cd /root/private_data/test
```

2. **运行训练脚本**:

```bash
python train.py
```

### 训练配置

训练参数在 `conf/transolver_rae2822.yaml` 中配置：

```yaml
model:
  name: 'Transolver'  # 模型名称
  specific_params:
    Transolver:
      n_hidden: 256      # 隐藏层维度
      n_layers: 8       # Transformer 层数
      space_dim: 7      # 空间特征维度
      fun_dim: 0        # 函数特征维度
      n_head: 8         # 多头注意力头数
      mlp_ratio: 2      # MLP 扩展比例
      out_dim: 4        # 输出维度
      slice_num: 32     # 物理切片数量
      unified_pos: 1    # 统一位置编码
      build_graph: True  # 构建图结构
      r: 0.05          # 邻域半径
      max_neighbors: 64  # 最大邻居数

training:
  max_epoch: 500       # 最大训练轮数
  lr: 0.001          # 学习率
  gpuid: 0           # GPU ID
  patience: 50        # 早停耐心值
  loss_criterion: 'MSE'  # 损失函数
  checkpoint_dir: './checkpoints/transolver_rae2822'  # 检查点目录
```

### 训练过程

训练脚本执行以下步骤：

1. **初始化分布式环境**: 使用 `DistributedManager`
2. **加载配置**: 从 YAML 文件加载模型和数据配置
3. **初始化数据管道**: 创建训练、验证和测试数据加载器
4. **初始化模型**: 根据配置创建 Transolver 模型
5. **训练循环**:
   - 前向传播
   - 计算 MSE 损失
   - 反向传播
   - 更新参数
   - 学习率调度（OneCycleLR）
6. **验证**: 每个 epoch 后在验证集上评估
7. **保存检查点**: 保存最佳验证损失的模型
8. **早停**: 如果验证损失在 `patience` 个 epoch 内没有改善，停止训练
9. **测试**: 训练结束后在测试集上评估

### 训练输出

训练过程中会生成以下输出：

- **日志信息**: 显示每个 epoch 的训练损失、验证损失、学习率等
- **检查点**: 保存在 `./checkpoints/transolver_rae2822/Transolver.pth`
- **归一化统计量**: 保存在 `./dataset/` 目录
  - `mean_in.npy`, `std_in.npy`
  - `mean_out.npy`, `std_out.npy`
- **测试结果**: 保存在检查点目录
  - `coef_norm.npy`: 归一化系数
  - `test_losses.npy`: 测试损失

### 训练监控

训练脚本使用 `tqdm` 显示进度条，实时显示：

- 当前 epoch 和总 epoch 数
- 当前 batch 的损失
- 当前学习率

每个 epoch 结束后，会输出：

- 训练损失
- 验证损失
- 训练时间
- 是否保存了新的最佳检查点

## 推理流程

### 推理步骤

1. **进入项目目录**:

```bash
cd /root/private_data/test
```

2. **运行推理脚本**:

```bash
python inference.py
```

### 推理配置

推理脚本使用与训练相同的配置文件 `conf/transolver_rae2822.yaml`，确保模型参数和数据设置一致。

### 推理过程

推理脚本执行以下步骤：

1. **加载配置**: 从 YAML 文件加载配置
2. **初始化数据管道**: 创建测试数据加载器
3. **加载模型**:
   - 从检查点目录加载最佳模型
   - 设置为评估模式 (`model.eval()`)
4. **推理循环**:
   - 遍历测试数据集
   - 前向传播（不计算梯度）
   - 计算预测损失
   - 收集预测和目标
5. **计算指标**:
   - 平均损失
   - 每个输出通道的损失
   - 损失的标准差
6. **保存结果**:
   - 预测结果
   - 目标值
   - 归一化系数
   - 评估指标

### 推理输出

推理结果保存在 `./results/rae2822/` 目录：

- **score_Transolver.json**: 评估指标摘要
  ```json
  {
    "model_name": "Transolver",
    "mean_loss": 0.001234,
    "std_loss": 0.000567,
    "channel_losses": [0.001, 0.002, 0.001, 0.000],
    "n_samples": 100
  }
  ```

- **predictions.npy**: 预测结果数组 `[n_samples, n_points, 4]`
- **targets.npy**: 目标值数组 `[n_samples, n_points, 4]`
- **mean_in.npy**: 输入均值
- **std_in.npy**: 输入标准差
- **mean_out.npy**: 输出均值
- **std_out.npy**: 输出标准差

### 推理监控

推理脚本使用 `tqdm` 显示进度条，实时显示推理进度。

推理完成后，会输出：

- 平均损失
- 损失标准差
- 各通道损失

## 项目结构

```
/root/private_data/test/
├── RAE2822.py                      # 数据集接口（智能体生成）
├── __init__.py                     # 包初始化（智能体生成）
├── train.py                        # 训练脚本（智能体生成）
├── inference.py                    # 推理脚本（智能体生成）
├── conf/
│   └── transolver_rae2822.yaml    # 配置文件（智能体生成）
├── dataset/                       # 归一化统计量（训练后生成）
│   ├── mean_in.npy
│   ├── std_in.npy
│   ├── mean_out.npy
│   └── std_out.npy
├── checkpoints/                    # 模型检查点（训练后生成）
│   └── Transolver.pth
└── results/                       # 推理结果（推理后生成）
    └── rae2822/
        ├── score_Transolver.json
        ├── predictions.npy
        ├── targets.npy
        ├── mean_in.npy
        ├── std_in.npy
        ├── mean_out.npy
        └── std_out.npy
```

## 代码审查结果

智能体在生成完成后自动进行了代码审查，检查了以下项目：

### ✅ 已检查项目

- datapipe 文件名和类名正确
- 继承 `BaseDataset`
- 返回 PyG Data 对象
- 输入输出维度与 config 一致（7 维输入，4 维输出）
- 归一化统计量只在训练集计算
- train.py 和 inference.py 导入路径正确
- checkpoint 加载和保存逻辑一致
- 使用 `model.eval()` 和 `torch.no_grad()`
- `test_dataloader()` 正确解包元组
- 归一化系数正确保存为独立文件

### ✅ 已修复问题

1. **导入路径问题**: 修复了 train.py 和 inference.py 中的导入路径
   - 从 `from onescience.datapipes.cfd import RAE2822Datapipe`
   - 改为 `from RAE2822 import RAE2822Datapipe`

2. **dataloader 解包问题**: 修复了 test_dataloader() 的调用
   - 正确解包元组：`test_loader, _ = datapipe.test_dataloader()`

3. **归一化系数保存问题**: 修复了 coef_norm 的保存
   - 分别保存为四个独立文件：mean_in, std_in, mean_out, std_out

4. **训练脚本测试部分**: 简化了测试逻辑，移除了对 AirfRANS 专用指标的依赖

## 注意事项

### 数据集适配差异

与 AirfRANS 数据集相比，RAE2822 数据集有以下差异：

1. **数据格式**: RAE2822 使用规则网格数据（npy 格式），AirfRANS 使用非结构化网格（vtu/vtp 格式）
2. **表面点**: RAE2822 数据集中没有明确的表面点信息，当前实现中 `surf` mask 全为 0
3. **法向量**: 法向量简化为 0，可能影响模型性能
4. **湍流粘度**: RAE2822 数据集中没有湍流粘度数据，输出中的 `nu` 通道设为 0

### 性能优化建议

如果需要更好的模型性能，可以考虑：

1. **计算真实法向量**: 从坐标数据中计算真实的表面法向量
2. **识别表面点**: 根据翼型几何判断哪些点是表面点
3. **添加湍流模型**: 如果有湍流数据，可以添加到输出中
4. **调整损失函数**: 使用加权损失，给表面点更高的权重

### 分布式训练

训练脚本支持分布式训练，使用以下命令启动：

```bash
# 单 GPU
python train.py

# 多 GPU（使用 torchrun）
torchrun --nproc_per_node=4 train.py
```

## 参考资源

### OneScience 框架

- [OneScience GitHub](https://github.com/onescience/onescience)
- [OneScience 文档](https://docs.onescience.ai)

### 数据集论文

- "A comparative study of learning techniques for the compressible aerodynamics over a transonic RAE2822 airfoil"
- DOI: [10.1016/j.compfluid.2022.105759](https://doi.org/10.1016/j.compfluid.2022.105759)

### Transolver 模型

- Transolver: A Transformer-based Solver for PDEs
- 参考: `onescience/src/onescience/models/transolver/`

## 联系方式

如有问题或建议，请：

1. 查看 OneScience 文档
2. 检查 oneskills 的生成日志
3. 联系 OneScience 开发团队

## 许可证

本案例遵循 Apache 2.0 许可证，与 RAE2822 数据集和 OneScience 框架保持一致。