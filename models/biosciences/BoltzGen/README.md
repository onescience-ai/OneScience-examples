<p align="center">
  <strong>
    <span style="font-size: 30px;">BoltzGen</span>
  </strong>
</p>

# 模型介绍

BoltzGen 是一个面向生物分子结合物设计的生成式模型。它能够根据蛋白质、肽、核酸或小分子靶标及约束条件生成候选结构，再通过逆折叠生成序列，使用 Boltz-2 完成结构复折叠、置信度分析、筛选和排序。



官方项目：[HannesStark/boltzgen](https://github.com/HannesStark/boltzgen)

# 模型描述

完整设计流水线包含六个阶段：

1. `design`：扩散模型生成满足目标和约束的候选三维骨架；
2. `inverse_folding`：逆折叠模型为候选骨架生成氨基酸序列；
3. `folding`：Boltz-2 对设计序列进行结构预测；
4. `design_folding`：在设计条件下再次折叠，用于检查条件遵循情况；
5. `analysis`：计算 RMSD、置信度、序列组成及其他质量指标；
6. `filtering`：按阈值筛选、排序，并生成结构文件、CSV 和汇总 PDF。

其中 `boltzgen1_diverse.ckpt` 和 `boltzgen1_adherence.ckpt` 是两种设计扩散模型，`boltzgen1_ifold.ckpt` 用于逆折叠，`boltz2_conf_final.ckpt` 用于结构预测；涉及蛋白质—小分子亲和力任务时还会使用 `boltz2_aff.ckpt`。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 蛋白质 binder 设计 | 针对给定靶标生成蛋白质结合物骨架和序列 |
| 肽与环肽设计 | 生成线性肽、环肽、二硫键肽等候选结构 |
| 抗体与纳米抗体设计 | 使用框架、CDR 或结合位点约束生成候选设计 |
| 小分子结合设计 | 针对小分子靶标设计结合蛋白，并可计算亲和力相关指标 |
| 逆折叠 | 为给定蛋白质骨架生成可能的氨基酸序列 |
| 训练链路验证 | 使用官方训练入口检查数据加载、前向、损失、反向传播和参数更新 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 支持 OneScience DTK 环境中的 DCU；
- DTK/HIP 设备通过 PyTorch 的 `torch.cuda` 兼容接口访问；
- 完整设计流水线建议在 DCU 上运行，CPU 目前仅适合导入、配置和轻量数据检查。

### 下载模型包

获取本适配模型包并解压后，进入仓库的相对目录：

```bash
python -m pip install modelscope
modelscope download --model OneScience/BoltzGen --local_dir ./BoltzGen
cd BoltzGen
```



### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
python -m pip install "onescience[bio-dcu]" \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

在 OneScience 环境基础上安装 BoltzGen 的额外依赖，BoltzGen 的 `bf16-mixed` 配置不能由旧版 `pytorch-lightning==1.8.6` 解析，因此对该依赖进行了升级`pytorch-lightning==2.5.6`：

```bash
python -m pip install --no-deps -r requirements.txt
```
验证 BoltzGen 命令行入口能否正常导入。
```bash
python scripts/boltzgen.py --help
```





### 权重与分子字典准备

将官方 checkpoint 放在 `weight/` 中：

| 相对位置 | 用途 |
| --- | --- |
| `weight/boltzgen1_diverse.ckpt` | 偏多样性的 binder 骨架扩散模型 |
| `weight/boltzgen1_adherence.ckpt` | 偏条件遵循的 binder 骨架扩散模型 |
| `weight/boltzgen1_ifold.ckpt` | 逆折叠序列生成模型 |
| `weight/boltz2_conf_final.ckpt` | Boltz-2 结构折叠与置信度模型 |
| `weight/boltz2_aff.ckpt` | 蛋白质—小分子任务的亲和力模型 |
| `weight/mols` | 推理所需的CCD分子字典 |


### DCU 最小推理

**作用：** 使用官方 `1g13` 示例和本地 checkpoint，运行一个候选的蛋白质 binder 完整设计流水线。

先设置离线模式和单卡 DCU：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

```

离线变量阻止计算节点访问外网。

生成流水线配置：

```bash
python scripts/boltzgen.py configure \
  conf/example/vanilla_protein/1g13prot.yaml \
  --output output/dcu_minimal \
  --protocol protein-anything \
  --num_designs 1 \
  --budget 1 \
  --devices 1 \
  --num_workers 0 \
  --use_kernels false \
  --moldir weight/mols \
  --design_checkpoints \
    weight/boltzgen1_diverse.ckpt \
    weight/boltzgen1_adherence.ckpt \
  --inverse_fold_checkpoint weight/boltzgen1_ifold.ckpt \
  --folding_checkpoint weight/boltz2_conf_final.ckpt
```

该命令只生成 `output/dcu_minimal/config/` 下的分阶段配置，不执行模型计算。配置成功表示设计输入、权重路径和流水线参数能够被解析。

执行流水线：

```bash
python scripts/boltzgen.py execute output/dcu_minimal
```

该命令按已生成配置执行六个阶段。全部步骤退出码为 0 表示端到端工程链路可运行；最终候选是否具有设计价值，仍应依据筛选 CSV、结构质量指标和实验验证判断。

蛋白质—小分子协议还需要在配置命令中增加：

```text
--affinity_checkpoint weight/boltz2_aff.ckpt
```

该参数启用亲和力 checkpoint，所得数值用于模型内部评价和候选比较，不应直接解释为实验测得的结合常数。

### 分阶段恢复推理

**作用：** 当流水线中断或只需重跑某个阶段时，复用已有配置和中间结果，避免重新执行已经完成的计算。

例如仅重新执行最终筛选：

```bash
python scripts/boltzgen.py execute output/dcu_minimal --steps filtering
```

命令成功后会重建 `output/dcu_minimal/final_ranked_designs/` 中的排序表、候选结构和汇总 PDF。筛选成功不表示一定有候选通过默认阈值；应同时检查通过数量和具体失败指标。

### 推理输出

主要结果位于：

```text
output/dcu_minimal/
├── config/                         # 六个阶段的实际运行配置
├── design/                         # 扩散模型生成的候选骨架
├── inverse_folding/                # 逆折叠生成的序列
├── folding/                        # 序列复折叠结果
├── design_folding/                 # 条件复折叠结果
├── analysis/                       # 候选质量指标
└── final_ranked_designs/           # 排序结果、结构文件和 PDF 汇总
```

`final_ranked_designs` 用于查看候选排名和质量摘要；结构坐标无 NaN/Inf 只能说明数值输出有效，不能替代结构合理性、结合能力或湿实验验证。

### 最小训练链路

BoltzGen 官方提供 PyTorch Lightning 训练入口，可训练以下部分：

| 配置 | 被训练的模型 |
| --- | --- |
| `inverse_folding.yaml` | 逆折叠序列生成模型 |
| `boltzgen_small.yaml` | 小型 BoltzGen 结构扩散模型 |
| `boltzgen.yaml` | 大型 BoltzGen 结构扩散模型 |



运行准备好的最小训练配置：

```bash
python scripts/train.py conf/train_boltzgen_small_smoke.yaml
```

该命令调用官方训练入口。日志出现一个 batch 完成且退出码为 0，表示 DataLoader、特征构建、forward、loss、backward 和 `optimizer.step` 已连通；它只验证训练代码可运行，不代表模型已经收敛或获得可用精度。

正式训练需要官方格式的 `targets/structures`、`targets/records`、`manifest.json`、MSA 和 `mols`。共享 OpenFold 旧结构 NPZ 需要先完成字段兼容转换；训练数据正在准备中。



# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 官方实现：[HannesStark/boltzgen](https://github.com/HannesStark/boltzgen)
- 本项目上游代码采用 MIT License，详见根目录 `LICENSE`；模型权重、训练数据和第三方资产仍应遵守各自的许可证与使用条款。
