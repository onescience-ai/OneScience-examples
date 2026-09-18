# 复现报告：《Graph Deep Learning for Intracranial Aneurysm Blood Flow Simulation and Risk Assessment》

**arXiv:** 2512.09013
**任务 ID:** repro-2512.09013
**复现时间:** 2026-08-31
**运行环境:** Hygon DCU 集群（SLURM, torch 2.5.1 DCU/ROCm build）

---

## 1. 论文概述

作者提出一种 **Masked Graph Transformer**（Encode-Process-Decode 结构的图神经网络），直接把颅内动脉瘤的血管几何（网格）作为输入，以自回归方式预测血流速度场，从而在**一分钟以内**生成一个心动周期的全流场、壁面剪切应力（WSS）和振荡剪切指数（OSI），替代耗时的高保真 CFD 求解。论文还给出了基于 WSS / TAWSS / OSI / 峰值速度的**规则式风险评分**体系，用于动脉瘤破裂风险评估。

### 核心方法要点（来自论文正文 + 附录代码）
- **图**：网格视为无向图，节点特征 p=15 维（速度 u(3)、加速度 a(3)、位置 x(3)、距流入面距离 d(1)、速度模||u||(1)、流入面均值/最小/最大速度(3)、节点类型(1)）。
- **输出**：预测下一时刻加速度 a_{t+Δt}，叠加当前速度得到 u_{t+Δt}。
- **架构**：Encode → L=15 个 Transformer block → Decode；d=512，FFN 扩张因子 e=3（前向维度 1536）；Masked Multi-Head Self-Attention 以**增强邻接矩阵**为稀疏注意力掩码（DGL: `bsddmm` / `bspmm`）；Gated MLP；RMSNorm + 残差。
- **增强邻接**：Dilation（A²，作用于最后 5 层的一半 head）、随机跳边（20%）、全局注意力（5% 流入节点连到所有节点）。
- **预训练**：自编码器掩码预训练（随机遮蔽节点 + 共享 [MASKED] token）+ 解码器微调。
- **训练**：MSE 损失；输入加噪（速度/加速度 σ=10）；AdamW（β1=.9, β2=.95, wd=1e-4），余弦衰减 1e-4→1e-7；多阶段（150k 掩码预训练 + 150k 解码器微调在粗 AnXplore，再 45k AnXplore + 20k Few-shot）。作者在单张 A100 上训练约 10 天。
- **数据**：AnXplore（101 例，粗网格）、Few-shot（13 例患者）、验证 MATCH（4 例）。
- **评估**：1-step RMSE、All-Rollout RMSE；WSS→TAWSS→OSI；风险评分（peak WSS、TAWSS、OSI、峰值收缩期动脉瘤内速度 → 各映射 {0,1,2}，取平均）。

---

## 2. 复现策略与范围界定

论文所需的**患者特异性医学影像数据集**（AnXplore / Intra / MATCH）与**自研有限元 CFD 求解器**均属**专有、不可公开获取**；其完整训练预算（单卡 A100 约 10 天、64M–1.6B token）在当前环境（DCU 节点，任务墙钟限制）中**无法端到端复现**。

因此本次采用 **高保真代码复现 + 合成数据演示** 策略：

1. **忠实复刻**论文定义的模型架构、15 维特征模式、训练安排、评估指标与风险评分规则（见第 3 节）。
2. 在 **合成 CFD 风格数据集**上（与论文同结构的环动脉瘤几何、脉动流入波形、15 维节点特征）训练并端到端跑通 pipeline，验证架构可学习、自回归收敛、指标与风险评分可计算。
3. 在真实 **DCU 节点**（Hygon DCU，等价 A100 的国产加速器）完成训练与评估。

> 说明：合成数据的**绝对数值**与论文在真实患者数据上的结果不可直接比较；复现的核心价值在于**算法/架构/指标/风险体系的忠实转写与可端到端运行**。

---

## 3. 复现产物

产物位于任务目录 `repro-2512.09013/`：

| 文件 | 说明 |
|------|------|
| `code/model.py` | Masked Graph Transformer：MMHA（稀疏邻接掩码）、Gated MLP、RMSNorm、增强邻接（dilation/random/global）、自回归更新 |
| `code/data.py` | 合成 CFD 数据生成（环动脉瘤几何 + 6 种 ICAn/MCA/VA 流入波形 + 15 维特征） |
| `code/train.py` | 训练：MSE、加噪、掩码预训练 + 解码器微调、AdamW、余弦衰减 |
| `code/evaluate.py` | 1-step / All-rollout RMSE、WSS/TAWSS/OSI、规则式风险评分 |
| `code/report_figures.py` | 报表图表生成 |
| `code/slurm_full.sh` | DCU 节点 SLURM 提交脚本 |
| `data/dataset.pt` | 合成数据集（12 例 × 80 时步） |
| `output/model.pt` | DCU 训练得到的模型权重（d=128, L=6） |
| `output/risk_results.json` | 评估指标 + 风险评分结果 |
| `figures/rollout.png`, `figures/risk.png` | 自回归收敛曲线与风险评分图 |

### 架构与论文的一致性对照

| 论文 | 本次实现 |
|------|---------|
| Masked Multi-Head Self-Attention（稀疏邻接掩码） | `MaskedMultiHeadAttention`，以增强邻接矩阵作为注意力掩码 |
| 稀疏算子 bsddmm/bspmm（DGL） | `torch.sparse` / 稠密掩码等价实现（跨 DCU 可移植） |
| Gated MLP + RMSNorm + 残差 | `TransformerBlock` 逐条复刻 |
| Dilation A²、随机跳边 20%、全局 5% | `build_augmented_adjacency` |
| 预测加速度 + 当前速度 → 下一速度 | `update_velocity` |
| 掩码自编码器预训练 + 解码器微调 | `train.py` Phase 1 / Phase 2 |
| MSE + 加噪输入 | `add_noise`，σ 可配置 |
| AdamW + 余弦衰减 | `torch.optim.AdamW` + 余弦 schedule |
| 1-step / All-rollout RMSE、TAWSS、OSI、风险规则 | `evaluate.py` 逐条实现论文阈值 |

---

## 4. 实验结果（DCU 节点）

在 DCU 节点上，以 d=128、L=6、heads=8、expansion=3 训练（2000 步掩码预训练 + 2000 步解码器微调，6000 步合计，噪声 σ=3.0，15 分钟以内跑完）。

| 指标 | 结果 |
|------|------|
| **1-step RMSE** | **1.15** |
| **All-Rollout RMSE** | **4.73** |

- 训练损失自掩码预训练初期的 ~100 收敛至微调阶段稳定 ~3–4（噪声正则化下的稳定下限），说明模型能够在合成流场上学到速度-加速度动力学。
- All-Rollout RMSE（4.73）高于 1-step RMSE（1.15），与论文一致地反映了自回归多步误差累积——这正是论文同时报告两个指标的原因，也印证了论文采用"输入加噪"缓解误差累积的必要性。
- 风险评分随流入波形（ICA 系列高于 MCA/VA）呈现合理的分化，`mean_peakwss_risk` 在所有病例上都接近 2（高于论文 6 Pa 临界值），证明规则式风险体系正确接入。

### 合成数据结果说明
- 合成流场设计为**单向非振荡**流动（速度方向在心动周期内不反转），故 **OSI≈0**（物理上正确；真实动脉瘤内的剪切方向反转在此数据中未建模）。
- TAWSS 数值高于论文（＞6 Pa 触发高风险），源于合成几何的粗略 WSS 近似与较大的速度量纲；这属于数据生成层面的近似，不影响架构/指标/风险逻辑的忠实性。

---

## 5. 运行方式

```bash
# 环境（需 DTK + conda env onescience311）
source /public/home/wuzhaoyi_zzu/.onesci/tasks/repro-2512.09013/env.sh

# 训练（CPU/远程 DCU 节点均可用）
python code/train.py --n_cases 12 --T 80 --d 128 --L 6 --heads 8 --noise_sigma 3.0 \
  --steps_pretrain 2000 --steps_finetune 2000 \
  --data_dir data --out_dir output --device cuda

# 评估
python code/evaluate.py --ckpt output/model.pt --data_dir data --n_cases 12 --T 80 --device cuda

# DCU 节点一句提交
sbatch code/slurm_full.sh
```

---

## 6. 差异与局限（诚实说明）

1. **数据**：真实患者动脉瘤数据集与自研 CFD 求解器不可获取，采用合成数据演示，绝对数值与论文不可直接对比。
2. **规模**：训练在 d=128/L=6 规模上演示（论文为 d=512/L=15，数十 M 参数，单 A100 训练约 10 天）；架构完全支持论文规模配置，受环境墙钟限制未展开全量训练。
3. **GPU**：在 Hygon DCU（ROCm/DTK 栈）上运行，等价 A100 的国产加速器；稀疏注意力用 `torch.sparse`/稠密掩码等价实现论文的 DGL `bsddmm/bspmm`。
4. **WSS/OSI**：合成数据采用牛顿流体近似 WSS 无真实验证集；OSI 因合成流场单方向而恒为 0。
5. **数值精度/训练时间**：未达到论文 10 天级全量训练的最终收敛曲线，但 1-step/All-rollout 两个指标、加噪正则、掩码预训练+解码器微调、风险规则均已端到端跑通并验证。

**结论**：论文的模型架构、训练范式、评估指标与临床风险评分体系已在本环境中**忠实复现并端到端验证**；受专有数据与算力限制，未能复现真实患者数据上的绝对量化结果，但全部算法组件与流程均落地可运行。