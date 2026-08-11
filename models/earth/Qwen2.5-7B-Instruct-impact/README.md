# Qwen2.5-7B-Instruct-impact 验证示例

本目录用于验证 Hugging Face 模型
[`extreme-weather-impacts/Qwen2.5-7B-Instruct-impact`](https://huggingface.co/extreme-weather-impacts/Qwen2.5-7B-Instruct-impact)。该模型基于 Qwen2.5-7B-Instruct 微调，用于判断企业披露文本是否表明企业已经受到极端天气事件的实际影响。

## 模型信息

- 固定版本：`4aa57015775cd4feac38a2c9b564430d5dbff53e`
- 许可证：Apache-2.0
- 任务：极端天气影响二分类（`Yes` / `No`）
- 参数量：7,615,616,512
- 权重格式：4 个 safetensors 分片，总计约 15.23 GB
- 官方模型卡后端：vLLM
- 本次超算验证后端：Transformers（容器内 vLLM 与 NVML 不兼容）

## 目录内容

- `train.py`：完整性检查、测试数据发现/生成、推理和指标计算脚本。
- `download_model.sh`：下载固定版本的配置、分词器和四个权重分片，并逐文件检查大小与 SHA256。
- `model_manifest.json`：模型来源、固定版本和权重清单。
- `requirements.txt`：依赖说明；深度学习运行时由超算容器提供。

权重、测试报告和 `validation_results/` 不进入 Git 仓库。

## 获取模型资源

OneScience 超算 Notebook 中的公共模型目录为：

```text
/root/group_data/SDU-Test/Qwen2.5-7B-Instruct-impact
```

如公共资源不可用，可在具有网络访问的 Linux 环境中执行：

```bash
chmod +x download_model.sh
./download_model.sh /path/to/Qwen2.5-7B-Instruct-impact
```

脚本支持 `.part` 文件断点续传。已经存在且校验正确的文件会自动跳过；已有文件校验失败时，脚本会停止，不会静默覆盖。

## 运行验证

`train.py` 从脚本所在目录加载模型。若代码与公共模型资源分开存放，可以建立一个验证目录，并将模型文件软链接到该目录后运行；也可以把本目录代码复制到模型资源所在目录。

容器内 vLLM 0.6.2 在本次设备上因 NVML 不支持计算能力查询而无法初始化，因此使用 Transformers 后端：

```bash
env USE_TORCH=1 USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_TF=1 \
python train.py \
  --device cuda \
  --backend transformers \
  --verify full \
  --batch-size 1
```

如果没有提供本地 CSV/JSON/JSONL 测试数据，脚本会生成 12 条平衡的合成企业披露样例，并计算：

- Accuracy
- Macro F1
- 可解析答案比例
- 严格输出格式比例
- 单样本中位推理延迟
- 峰值 GPU 显存

## 本次验证结果

在 OneScience Notebook 容器（Python 3.10.12、PyTorch 2.4.1、CUDA）中，使用公共目录模型资源进行完整 SHA256 验证和 Transformers 推理，结果为：

```text
Status:               PASS
Validation level:     functional_only
Data source:          synthetic_disclosures
Weight verification: full
Examples:             12
Accuracy:             1.000000
Macro F1:             1.000000
Parsed answers:       1.000000
Strict output format: 1.000000
Median latency:       0.236 s/example
Peak GPU memory:      14587.82 MB
```

这些指标仅证明模型文件完整、推理链路可运行，并能正确处理本脚本构造的功能性样例。由于没有使用官方独立真实测试集，本结果不能作为模型科学精度或泛化性能结论。

## 输入数据格式

可通过 `--data` 指定 CSV、JSON 或 JSONL 文件。记录需要包含文本字段和二分类标签字段；脚本会尝试识别常见字段名。建议使用明确的 `text` 和 `label` 字段，其中标签为 `Yes` 或 `No`。

示例：

```csv
id,text,label
1,"Flooding damaged our warehouse and halted shipments.",Yes
2,"Flooding may affect our operations in the future.",No
```

## 注意事项

- 完整 SHA256 校验需要顺序读取约 15 GB 权重，启动前会花费额外时间。
- 单卡 FP16 推理峰值显存约 14.6 GB，建议至少准备 16 GB 可用 GPU 显存。
- `Scientific accuracy: unavailable` 是预期结果，因为默认数据为合成功能样例。
