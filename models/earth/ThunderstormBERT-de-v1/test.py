import os
import json
import time
import platform

MODEL_DIR = "/root/private_data/ThunderstormBERT-de-v1"

required_files = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spm.model",
]

print("模型目录：", MODEL_DIR)
print("=" * 70)

missing_files = []

for filename in required_files:
    path = os.path.join(MODEL_DIR, filename)

    if os.path.isfile(path):
        size = os.path.getsize(path)
        print(f"{filename:<25} 存在，大小：{size / 1024 / 1024:.2f} MB")
    else:
        print(f"{filename:<25} 不存在")
        missing_files.append(filename)

if missing_files:
    raise FileNotFoundError(f"缺少文件：{missing_files}")

weight_path = os.path.join(MODEL_DIR, "model.safetensors")
weight_size = os.path.getsize(weight_path)

if weight_size < 1_000_000_000:
    raise RuntimeError(
        f"model.safetensors 只有 {weight_size / 1024 / 1024:.2f} MB，"
        "可能没有上传完整。正常大小约为 1063.61 MB。"
    )

print("=" * 70)
print("模型文件检查通过！")










import os

# 本模型只使用 PyTorch，禁止 Transformers 加载 TensorFlow 和 JAX
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

print("Python版本：", platform.python_version())
print("PyTorch版本：", torch.__version__)
print("Transformers版本：", transformers.__version__)
print("GPU是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("加速卡名称：", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("使用设备：", device)













if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_fast=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_safetensors=True,
)

model = model.to(device)
model.eval()

if torch.cuda.is_available():
    torch.cuda.synchronize()

load_time = time.perf_counter() - load_start

print("模型加载成功！")
print("模型类型：", model.__class__.__name__)
print("分词器类型：", tokenizer.__class__.__name__)
print("模型加载时间：", round(load_time, 4), "秒")
print("分类数量：", model.config.num_labels)
print("标签配置：", model.config.id2label)













test_texts = [
    "Am Nachmittag bildeten sich schwere Gewitter mit Blitz, Donner und Starkregen.",
    "Der Himmel war wolkenlos, und den ganzen Tag schien die Sonne.",
    "Ein heftiges Gewitter brachte Hagel, starken Regen und schwere Windböen.",
]

inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

print("测试文本数量：", len(test_texts))
print("input_ids形状：", tuple(inputs["input_ids"].shape))

for index, text in enumerate(test_texts, start=1):
    print(f"\n文本{index}：{text}")













# 先执行一次预热，不计入正式推理时间
with torch.no_grad():
    warmup_inputs = {
        name: tensor[:1]
        for name, tensor in inputs.items()
    }
    _ = model(**warmup_inputs)

if torch.cuda.is_available():
    torch.cuda.synchronize()

inference_start = time.perf_counter()

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)

if torch.cuda.is_available():
    torch.cuda.synchronize()

inference_time = time.perf_counter() - inference_start

print("模型推理完成！")
print("输出张量形状：", tuple(outputs.logits.shape))
print("推理时间：", round(inference_time, 6), "秒")








probabilities_cpu = probabilities.detach().cpu()

all_predictions = []

for text_index, text in enumerate(test_texts):
    print("=" * 80)
    print("输入文本：", text)
    print("预测结果：")

    scores = probabilities_cpu[text_index]
    sorted_indices = torch.argsort(scores, descending=True)

    text_predictions = []

    for rank, label_index in enumerate(sorted_indices.tolist(), start=1):
        label = model.config.id2label.get(
            label_index,
            f"LABEL_{label_index}"
        )
        probability = float(scores[label_index])

        print(
            f"第{rank}名：{label:<10} "
            f"概率：{probability:.6f}"
        )

        text_predictions.append({
            "rank": rank,
            "label": label,
            "probability": probability,
        })

    all_predictions.append({
        "text": text,
        "predictions": text_predictions,
    })

print("=" * 80)











result = {
    "status": "SUCCESS",
    "model": "Stickmu/ThunderstormBERT-de-v1",
    "model_directory": MODEL_DIR,
    "model_class": model.__class__.__name__,
    "tokenizer_class": tokenizer.__class__.__name__,
    "device": str(device),
    "accelerator_name": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    },
    "timing_seconds": {
        "model_load": round(load_time, 6),
        "inference_after_warmup": round(inference_time, 6),
    },
    "output_shape": list(outputs.logits.shape),
    "results": all_predictions,
}

result_path = os.path.join(
    MODEL_DIR,
    "test_result.json"
)

with open(result_path, "w", encoding="utf-8") as file:
    json.dump(
        result,
        file,
        ensure_ascii=False,
        indent=2,
    )

print("=" * 80)
print("测试状态：SUCCESS，模型可以正常运行")
print("模型名称：Stickmu/ThunderstormBERT-de-v1")
print("输出张量形状：", tuple(outputs.logits.shape))
print("模型加载时间：", round(load_time, 4), "秒")
print("推理时间：", round(inference_time, 6), "秒")
print("测试结果已保存到：", result_path)
print("=" * 80)