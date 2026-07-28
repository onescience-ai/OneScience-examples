import os
import random
import requests
import torch
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
model_path = "./"
try:
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    print("成功加载本地模型与预处理器！")
except Exception:
    model_path = "8kkillian/autotrain-weather-classification-3723199086"
    print(f"本地未找到模型，将从 HuggingFace 自动加载: {model_path}")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)

model.eval()
id2label = model.config.id2label
model_labels = list(id2label.values())
print(f"模型能够识别的类别 ({len(model_labels)} 类): {model_labels}\n")
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
        "https://images.unsplash.com/photo-1511289081-d06dda19034d?w=400",  # 蓝黑紫调强闪电
        "https://images.unsplash.com/photo-1429552077091-836152271555?w=400"   # 旷野一道闪电
    ],
    "rainbow": [
        "https://images.unsplash.com/photo-1508624217470-5ef0f947d8be?w=400",  # 原野彩虹
        "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=400",  # 蓝天清晰彩虹
        "https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?w=400"   # 晴空双彩虹
    ]
}

headers = {'User-Agent': 'Mozilla/5.0'}
cached_images_pool = {}

print("正在下载并预载多图样本集，请稍候...")
for cat_key, url_list in multi_weather_urls.items():
    # 自动对齐模型标准的类别名称大小写
    matched_label = next((l for l in model_labels if l.lower() == cat_key.lower()), cat_key)
    cached_images_pool[matched_label] = []
    
    for idx, url in enumerate(url_list):
        try:
            resp = requests.get(url, headers=headers, timeout=10, stream=True)
            if resp.status_code == 200:
                img = Image.open(resp.raw).convert("RGB")
                cached_images_pool[matched_label].append(img)
            else:
                placeholder = Image.new("RGB", (224, 224), color=(128, 128, 128))
                cached_images_pool[matched_label].append(placeholder)
        except Exception as e:
            print(f"  [{cat_key}] 第 {idx+1} 张图片下载失败: {e}")
            placeholder = Image.new("RGB", (224, 224), color=(128, 128, 128))
            cached_images_pool[matched_label].append(placeholder)

print("所有素材准备就绪！\n")
random.seed(42)  # 保证测试可复现
total_samples = 50
true_labels = []
pred_labels = []

categories = list(cached_images_pool.keys())

print(f"开始运行 {total_samples} 张多图混合推理评估...")
with torch.no_grad():
    for i in range(total_samples):
        # 轮流选择天气类别
        target_label = categories[i % len(categories)]
        
        # 从该类别的多张真实图中随机挑选 1 张（增加场景丰富度）
        img_candidates = cached_images_pool[target_label]
        selected_img = random.choice(img_candidates)
        
        # 推理
        inputs = processor(images=selected_img, return_tensors="pt")
        outputs = model(**inputs)
        pred_idx = outputs.logits.argmax(-1).item()
        pred_label = id2label[pred_idx]
        
        true_labels.append(str(target_label))
        pred_labels.append(str(pred_label))
        
        if (i + 1) % 10 == 0:
            print(f"   进度: 已完成 {i + 1} / {total_samples} 张评估")
            acc = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, zero_division=0)
labels_sorted = sorted(list(set(true_labels + pred_labels)))
cm = confusion_matrix(true_labels, pred_labels, labels=labels_sorted)

print("\n" + "="*65)
print(f"多图混合测试评估报告 (测试样本总数: {len(true_labels)})")
print("="*65)
print(f"整体准确率 (Overall Accuracy): {acc:.4f} ({acc*100:.2f}%)\n")

print("详细分类指标 (Precision / Recall / F1-Score):")
print(report)

print("混淆矩阵 (Confusion Matrix):")
cm_df = pd.DataFrame(
    cm, 
    index=[f"真实:{l}" for l in labels_sorted], 
    columns=[f"预测:{l}" for l in labels_sorted]
)
print(cm_df)
print("="*65)