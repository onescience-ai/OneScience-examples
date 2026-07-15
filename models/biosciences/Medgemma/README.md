
<p align="center">
  <strong>
    <span style="font-size: 30px;">MedGemma</span>
  </strong>
</p>

# 模型介绍

MedGemma 是 Google 开源的医学多模态大语言模型，基于 [Gemma 3](https://ai.google.dev/gemma/docs/core) 架构，针对医学文本与医学影像理解进行训练。MedGemma 提供两种变体：

- **MedGemma 4B**：多模态模型，支持医学文本与医学图像联合输入。
- **MedGemma 27B**：纯文本模型，专注于医学文本理解与问答。

MedGemma 4B 使用 [SigLIP](https://arxiv.org/abs/2303.15343) 图像编码器，已在多种去标识化医学数据上预训练，包括胸片（CXR）、皮肤科图像、眼科图像和组织病理学切片；其语言模型组件在放射学图像、病理学图像、眼科图像、皮肤科图像和医学文本上进行了训练。

MedGemma 已在多项临床相关基准上评估，涵盖开放基准数据集和专家人工评估任务。更多信息请参阅：

- [开发者文档](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
- [模型卡（Model Card）](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [Hugging Face 模型](https://huggingface.co/models?other=medgemma)
- [Google Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)

当前示例默认基于 `google/medgemma-1.5-4b-it`（4B 多模态指令模型），支持文本与医学图像联合输入，可在 GPU 或海光 DCU 平台上运行。

# 仓库说明

本示例将 MedGemma 集成到 OneScience 生物信息（AI for Biology）组件中，提供面向医学场景的统一推理、微调与评估入口。

当前支持能力：

- **医学问答**：基于 MedQA 等医学知识基准评估模型问答能力。
- **医学影像分析**：支持胸片（CXR）解剖结构定位、多期影像对比分析等任务。
- **领域微调**：基于 NCT 结肠组织病理图像等数据，使用 LoRA 进行参数高效微调。
- **统一推理接口**：通过 `MedicalInferenceRunner` 提供交互式与批量文件推理能力。

当前不支持能力：

- 原 README 未说明当前不支持的能力。

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 医学问答 | 基于 MedQA 等医学知识基准评估模型问答能力。 |
| 医学影像分析 | 支持胸片（CXR）解剖结构定位、多期影像对比分析等任务。 |
| 领域微调 | 基于 NCT 结肠组织病理图像等数据，使用 LoRA 进行参数高效微调。 |
| 统一推理 | 通过 `MedicalInferenceRunner` 提供交互式与批量文件推理能力。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configs/inference_config.yaml` | 推理配置示例 | 配置模型与推理参数 |
| `configs/configs_base.py` | 基础配置定义 | - |
| `runner/medical_inference_runner.py` | 统一医学推理运行器 | 支持交互式与批量推理 |
| `scripts/run_evaluate_on_medqa.sh` | MedQA 医学问答评估 | 输出至 `scripts/medqa_results/` |
| `scripts/run_cxr_anatomy.sh` | 胸片解剖结构定位 | 输出至 `scripts/outputs/` |
| `scripts/run_cxr_longitudinal_comparison.sh` | 胸片前后对比分析 | 输出至 `scripts/compare_outputs/` |
| `scripts/run_fine_tune.sh` | 病理图像 LoRA 微调 | 输出至 `scripts/medgemma-nct-lora/` |
| `tests/test_integration.py` | 集成测试脚本 | 验证模块与组件导入 |

```
medgemma/
├── configs/                          # 配置目录
│   ├── inference_config.yaml         # 推理配置示例
│   └── configs_base.py               # 基础配置定义
├── runner/
│   └── medical_inference_runner.py   # 统一医学推理运行器
├── scripts/                          # 可执行脚本
│   ├── notebook_conver/              # 脚本调用的 Python 实现
│   │   ├── cxr_anatomy_localization_with_hugging_face.py
│   │   ├── cxr_longitudinal_comparison.py
│   │   ├── evaluate_on_medqa.py
│   │   ├── fine_tune_with_hugging_face.py
│   │   └── detect_image_token.py
│   ├── run_cxr_anatomy.sh            # 胸片解剖结构定位（已提供）
│   ├── run_cxr_longitudinal_comparison.sh  # 胸片前后对比分析（已提供）
│   ├── run_evaluate_on_medqa.sh      # MedQA 医学问答评估（已提供）
│   └── run_fine_tune.sh              # 病理图像 LoRA 微调（已提供）
├── tests/
│   └── test_integration.py           # 集成测试脚本
└── README.md                         # 本文档
```

# 使用说明

## 1. OneCode使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用GPU或DCU运行。
- CPU可以用于连通性验证，但速度较慢。
- DCU用户需要预先安装DTK，建议使用DTK 25.04.2以上版本或与当前集群匹配的OneScience推荐版本。

**软件要求**

DCU用户想了解更多适配内容请联系 liubiao@sugon.com

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光DCU：

```bash
hy-smi
```

### 环境准备

1. 检查 `botocore` 的版本，若版本过低，请升级：

    ```bash
    pip install --upgrade boto3==1.43.36 botocore==1.43.36
    ```

2. 检查 `transformers` 版本，若版本过低，请升级：

    ```bash
    pip install --upgrade transformers==5.12.1
    ```



## 快速开始

### 1. 安装运行环境

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 2. 下载数据库(含权重)

```bash
modelscope download --dataset OneScience/medgemma --local_dir ./dataset
```

脚本默认从以下路径加载模型：

```
${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it
```

脚本默认从以下路径加载数据集：

| 任务 | 数据 | 默认路径 |
|------|------|----------|
| MedQA 评估 | MedQA parquet 数据 | `${ONESCIENCE_DATASETS_DIR}/medgemma/medqa` |
| 胸片解剖定位 | 胸片图像 | `${ONESCIENCE_DATASETS_DIR}/medgemma/Chest_Xray/...` |
| 胸片纵向对比 | 前后两次胸片 | `${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/...` |
| 病理图像微调 | NCT-CRC-HE-100K / CRC-VAL-HE-7K | `${ONESCIENCE_DATASETS_DIR}/medgemma/nct/...` |

请提前下载模型并放置到该目录，或通过 `model_path` 环境变量覆盖。

### 3. 使用方式

#### 1. 集成测试

验证 MedGemma 在 OneScience 中的模块、配置、数据适配器与图像处理组件是否可正常导入：

```bash
cd examples/biosciences/medgemma
python tests/test_integration.py
```
---

#### 2. 医学问答评估（`run_evaluate_on_medqa.sh`）

在 MedQA 数据集上评估模型医学问答能力，默认处理 10 条样本用于快速验证。

```bash
cd examples/biosciences/medgemma
bash scripts/run_evaluate_on_medqa.sh
```

输出：

- `scripts/medqa_results/medqa_results.json`：每条样本的详细结果
- `scripts/medqa_results/summary.txt`：准确率等汇总指标

---

#### 3. 胸片解剖结构定位（`run_cxr_anatomy.sh`）

对单张或多张胸片进行解剖部位定位。脚本内部同时运行单图模式和批量模式：

```bash
cd examples/biosciences/medgemma
bash scripts/run_cxr_anatomy.sh
```

输出：

- `scripts/outputs/result_*.json`：定位坐标与标签
- `scripts/outputs/result_*.png`：带边界框标注的可视化图像
- `scripts/outputs/batch_summary.json`：批量模式汇总结果

---

#### 4. 胸片纵向对比分析（`run_cxr_longitudinal_comparison.sh`）

对同一患者的前后两次胸片进行对比分析：

```bash
cd examples/biosciences/medgemma
bash scripts/run_cxr_longitudinal_comparison.sh
```

输出：

- `scripts/compare_outputs/compare_<image1>_vs_<image2>.txt`：文本对比报告
- `scripts/compare_outputs/compare_<image1>_vs_<image2>.json`：结构化 JSON 结果

---

#### 5. 病理图像 LoRA 微调（`run_fine_tune.sh`）

基于 NCT 结肠组织病理图像数据集进行 LoRA 微调：

```bash
cd examples/biosciences/medgemma
bash scripts/run_fine_tune.sh
```

输出：

- `scripts/medgemma-nct-lora/`：LoRA 权重、训练日志与评估结果

> 注：脚本会自动检查并修复 `boto3==1.43.36` 和 `botocore==1.43.36` 版本，避免依赖冲突。

---

#### 6. 使用推理运行器

`runner/medical_inference_runner.py` 提供统一的推理入口，支持交互式与批量文件推理。

##### 交互式推理

```bash
cd examples/biosciences/medgemma
export PYTHONPATH=../../../src:$PYTHONPATH
python runner/medical_inference_runner.py \
    --config configs/inference_config.yaml \
    --interactive
```

##### 批量文件推理

```bash
cd examples/biosciences/medgemma
export PYTHONPATH=../../../src:$PYTHONPATH
python runner/medical_inference_runner.py \
    --config configs/inference_config.yaml \
    --input data/example_input.json
```
### 注意事项

- 运行脚本前需确保 `ONESCIENCE_DATASETS_DIR` 环境变量已正确设置。
- 脚本默认使用 `HIP_VISIBLE_DEVICES=0`，在海光 DCU 平台可直接运行；在 CUDA 平台可替换为 `CUDA_VISIBLE_DEVICES=0` 或根据设备调整。
- 如需使用 vLLM 加速推理，请确保已安装对应版本的 vLLM 并配置 `use_vllm: true`。
- 胸片解剖定位与病理微调脚本会自动修复 `boto3` / `botocore` 版本，避免依赖冲突。
- 4B 多模态模型推理显存需求较大，建议至少单卡 24GB 显存；多卡可通过 `num_gpus` 或外部 `CUDA_VISIBLE_DEVICES` 控制。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

MedGemma 模型采用 [Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms) 许可，本仓库示例代码采用 Apache 2.0 许可。

更多信息请参阅：

- [开发者文档](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
- [模型卡](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [社区准则](https://developers.google.com/health-ai-developer-foundations/community-guidelines)
- [Hugging Face](https://huggingface.co/models?other=medgemma)
- [Google Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)
