import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["USE_FLAX"] = "0"
os.environ["USE_JAX"] = "0"

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

print("模型目录：", MODEL_DIR)


from pathlib import Path
import torch
import transformers

model_path = Path(MODEL_DIR)

required_files = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

for filename in required_files:
    path = model_path / filename
    print(filename, "存在：" if path.exists() else "缺失！", path)

print("\nPyTorch版本：", torch.__version__)
print("Transformers版本：", transformers.__version__)
print("GPU是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("加速卡名称：", torch.cuda.get_device_name(0))




    import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    model_max_length=512,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model = model.to(device)
model.eval()

load_time = time.time() - start_time

print("模型加载成功！")
print("模型类型：", type(model).__name__)
print("加载时间：", round(load_time, 4), "秒")
print("分类数量：", model.config.num_labels)
print("标签：", model.config.id2label)




texts = [
    "Hurricanes play a significant role in our yearly risk assessment.",
    "Droughts increase the risk of severe wildfires that can additionally damage our crops.",
    "The company opened a new office and hired more employees."
]

inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)

inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

if device.type == "cuda":
    torch.cuda.synchronize()

start_time = time.time()

with torch.inference_mode():
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)

if device.type == "cuda":
    torch.cuda.synchronize()

inference_time = time.time() - start_time

id2label = {
    int(key): value
    for key, value in model.config.id2label.items()
}

for index, text in enumerate(texts):
    scores = probabilities[index].detach().cpu()

    ranked_results = sorted(
        [
            {
                "label": id2label[label_id],
                "probability": float(scores[label_id])
            }
            for label_id in range(len(scores))
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )

    print("=" * 80)
    print("输入文本：", text)
    print("预测结果：")

    for result in ranked_results:
        print(
            f"{result['label']:10s} "
            f"{result['probability']:.6f}"
        )

print("=" * 80)
print("输出张量形状：", tuple(outputs.logits.shape))
print("推理时间：", round(inference_time, 6), "秒")
print("测试状态：SUCCESS，模型可以正常运行")