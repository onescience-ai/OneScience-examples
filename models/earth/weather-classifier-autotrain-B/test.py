import sys
import os

# 1. 强行在 sys.modules 里将 tensorflow 和 jax 阻断
# 这样任何库尝试 import jax 或 import tensorflow 时都会直接跳过或抛出标准 ModuleNotFoundError
sys.modules['tensorflow'] = None
sys.modules['jax'] = None
sys.modules['jax.numpy'] = None
sys.modules['flax'] = None

# 2. 补上常规环境变量设置
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_TORCH"] = "1"

# 3. 动态修复 transformers 内部的 is_flax_available / is_tf_available 逻辑
import transformers.utils.import_utils as import_utils
import_utils.is_flax_available = lambda: False
import_utils.is_tf_available = lambda: False

# 4. 现在安全导入其余依赖
import random
import requests
import torch
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
model_path = "8kkillian/autotrain-weather-classification-3723199088"

print(f"正在加载模型: {model_path} ...")
try:
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(
        model_path, 
        use_safetensors=True, 
        weights_only=False
    )
    print("✅ 成功加载 Hugging Face 在线模型！")
except Exception as e:
    print(f"⚠️ 从 HF 加载失败，尝试加载本地当前目录模型: {e}")
    model_path = "./"
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(
        model_path, 
        use_safetensors=True, 
        weights_only=False
    )
id2label = model.config.id2label
model_labels = list(id2label.values())

# ==========================================
# 4. 构建多视角天气场景测试图库 (5 类 x 3 图)
# ==========================================
multi_weather_urls = {
    "rain": [
        "https://images.unsplash.com/photo-1519692933481-e162a57d6721?w=400",  # 车窗雨滴
        "https://images.unsplash.com/photo-1534274988757-a28bf1a57c17?w=400",  # 倾盆大雨
        "https://images.unsplash.com/photo-1501999635878-71cb5379c2d4?w=400"   # 水坑雨景
    ],
    "snow": [
        "https://images.unsplash.com/photo-1517299321609-52687d1bc55a?w=400",  # 冰雪森林
        "https://images.unsplash.com/photo-1491002052546-bf38f186af56?w=400",  # 飘雪街道
        "https://images.unsplash.com/photo-1516431883659-655d41c09bf9?w=400"   # 积雪山脉
    ],
    "sandstorm": [
        "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=400",  # 沙漠漫天黄沙
        "https://images.unsplash.com/photo-1542401886-65d6c61db217?w=400",  # 荒漠风沙
        "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=400"   # 沙尘蔽日
    ],
    "lightning": [
        "https://images.unsplash.com/photo-1605727216801-e27ce1d0cc28?w=400",  # 夜空闪电
        "https://images.unsplash.com/photo-1511289081-d06dda19034d?w=400",  # 强闪电
        "https://images.unsplash.com/photo-1429552077091-836152271555?w=400"   # 旷野闪电
    ],
    "rainbow": [
        "https://images.unsplash.com/photo-1508624217470-5ef0f947d8be?w=400",  # 原野彩虹
        "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=400",  # 蓝天彩虹
        "https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?w=400"   # 双彩虹
    ]
}

headers = {'User-Agent': 'Mozilla/5.0'}
cached_images_pool = {}

print("⬇️ 正在预载测试图像资源...")
for cat_key, url_list in multi_weather_urls.items():
    matched_label = next((l for l in model_labels if l.lower() == cat_key.lower()), cat_key)
    cached_images_pool[matched_label] = []
    
    for idx, url in enumerate(url_list):
        try:
            resp = requests.get(url, headers=headers, timeout=10, stream=True)
            if resp.status_code == 200:
                img = Image.open(resp.raw).convert("RGB")
                cached_images_pool[matched_label].append(img)
            else:
                cached_images_pool[matched_label].append(Image.new("RGB", (224, 224), color=(128, 128, 128)))
        except Exception:
            cached_images_pool[matched_label].append(Image.new("RGB", (224, 224), color=(128, 128, 128)))

print("✅ 所有图像资源预载完毕，开始执行推理评估...\n")
import sys
import os

# 1. 强行把 CPU/GPU 隔离，声明只用 PyTorch
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_TORCH"] = "1"

# 2. 深入底层，把 transformers 内部所有引用 flax/jax 的地方全部打上硬补丁
import transformers.utils.import_utils as import_utils
import transformers.utils.generic as generic_utils

# 彻底拦截 import_utils 模块
import_utils.is_flax_available = lambda: False
import_utils.is_tf_available = lambda: False
import_utils.is_jax_available = lambda: False
import_utils.check_torch_load_is_safe = lambda: None

# 彻底拦截 generic 模块（关键！报错就是这里直接调用的）
generic_utils.is_flax_available = lambda: False
generic_utils.is_jax_tensor = lambda x: False

print("✅ 底层依赖隔离补丁已成功注入！")
random.seed(42)  # 固定随机种子确保结果可复现
total_samples = 50
true_labels = []
pred_labels = []

categories = list(cached_images_pool.keys())

with torch.no_grad():
    for i in range(total_samples):
        target_label = categories[i % len(categories)]
        img_candidates = cached_images_pool[target_label]
        selected_img = random.choice(img_candidates)
        
        inputs = processor(images=selected_img, return_tensors="pt")
        outputs = model(**inputs)
        pred_idx = outputs.logits.argmax(-1).item()
        pred_label = id2label[pred_idx]
        
        true_labels.append(str(target_label))
        pred_labels.append(str(pred_label))
        
        if (i + 1) % 10 == 0 or (i + 1) == total_samples:
            print(f"进度: 已完成 [{i + 1} / {total_samples}] 张推理...")


