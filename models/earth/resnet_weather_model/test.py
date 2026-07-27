import os
import json
import time
import platform

MODEL_DIR = "/root/private_data/resnet_weather_model"
IMAGE_PATH = os.path.join(MODEL_DIR, "snow_test.jpg")
WEIGHT_PATH = os.path.join(MODEL_DIR, "pytorch_model.bin")

required_files = [
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "snow_test.jpg",
]

print("模型目录：", MODEL_DIR)
print("=" * 70)

missing_files = []

for filename in required_files:
    path = os.path.join(MODEL_DIR, filename)

    if os.path.isfile(path):
        size = os.path.getsize(path)
        print(f"{filename:<28} 存在，大小：{size / 1024 / 1024:.2f} MB")
    else:
        print(f"{filename:<28} 不存在")
        missing_files.append(filename)

if missing_files:
    raise FileNotFoundError(f"缺少文件：{missing_files}")

weight_size = os.path.getsize(WEIGHT_PATH)

if weight_size < 90_000_000:
    raise RuntimeError(
        f"pytorch_model.bin只有{weight_size / 1024 / 1024:.2f} MB，"
        "可能没有上传完整。正常大小约为90.08 MB。"
    )

print("=" * 70)
print("模型和图片文件检查通过！")






import os

# 本模型只使用PyTorch
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers

from PIL import Image
from IPython.display import display

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

print("Python版本：", platform.python_version())
print("PyTorch版本：", torch.__version__)
print("Transformers版本：", transformers.__version__)
print("GPU是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("加速卡名称：", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")

print("使用设备：", device)






if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

load_start = time.perf_counter()

# 读取本地配置
config = AutoConfig.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

# 加载图片预处理器
processor = AutoImageProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

# 根据配置创建模型结构
model = AutoModelForImageClassification.from_config(config)

# 以weights_only方式读取PyTorch权重
state_dict = torch.load(
    WEIGHT_PATH,
    map_location="cpu",
    weights_only=True,
)

# 将权重载入模型
load_result = model.load_state_dict(
    state_dict,
    strict=True,
)

del state_dict

model = model.to(device)
model.eval()

if torch.cuda.is_available():
    torch.cuda.synchronize()

load_time = time.perf_counter() - load_start

print("模型加载成功！")
print("模型类型：", model.__class__.__name__)
print("图片处理器类型：", processor.__class__.__name__)
print("模型加载时间：", round(load_time, 4), "秒")
print("分类数量：", model.config.num_labels)
print("缺失权重：", load_result.missing_keys)
print("多余权重：", load_result.unexpected_keys)
print("标签：", model.config.id2label)







image = Image.open(IMAGE_PATH).convert("RGB")

print("图片路径：", IMAGE_PATH)
print("图片尺寸：", image.size)
print("图片模式：", image.mode)

display(image)





inputs = processor(
    images=image,
    return_tensors="pt",
)

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

print("预处理完成！")

for name, tensor in inputs.items():
    print(
        f"{name}：形状={tuple(tensor.shape)}，"
        f"数据类型={tensor.dtype}，"
        f"设备={tensor.device}"
    )












# 先预热一次，不计入正式推理时间
with torch.no_grad():
    _ = model(**inputs)

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






scores = probabilities[0].detach().cpu()
sorted_indices = torch.argsort(scores, descending=True)

all_predictions = []

print("=" * 80)
print("输入图片：", IMAGE_PATH)
print("预测结果：")

for rank, label_index in enumerate(sorted_indices.tolist(), start=1):
    label = model.config.id2label.get(
        label_index,
        f"LABEL_{label_index}"
    )
    probability = float(scores[label_index])

    print(
        f"第{rank:>2}名：{label:<12} "
        f"概率：{probability:.6f}"
    )

    all_predictions.append({
        "rank": rank,
        "label": label,
        "probability": probability,
    })

top1 = all_predictions[0]

print("=" * 80)
print("Top-1预测类别：", top1["label"])
print("Top-1预测概率：", round(top1["probability"], 6))







result = {
    "status": "SUCCESS",
    "model": "sallyanndelucia/resnet_weather_model",
    "model_directory": MODEL_DIR,
    "image": IMAGE_PATH,
    "model_class": model.__class__.__name__,
    "processor_class": processor.__class__.__name__,
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
    "top1": top1,
    "predictions": all_predictions,
}

result_path = os.path.join(
    MODEL_DIR,
    "test_result.json",
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
print("模型名称：sallyanndelucia/resnet_weather_model")
print("输入图片：", IMAGE_PATH)
print("Top-1类别：", top1["label"])
print("Top-1概率：", round(top1["probability"], 6))
print("输出张量形状：", tuple(outputs.logits.shape))
print("模型加载时间：", round(load_time, 4), "秒")
print("推理时间：", round(inference_time, 6), "秒")
print("测试结果已保存到：", result_path)
print("=" * 80)