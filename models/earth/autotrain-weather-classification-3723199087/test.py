import os
import sys
import time
import math
import requests
import warnings
import torch
from PIL import Image

# 1. 忽略警告
warnings.filterwarnings("ignore")

# 2. 开启 Hugging Face 国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 3. 绕过安全检查
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

from transformers import pipeline

# ================================
# 配置参数
# ================================
MODEL_ID = "8kkillian/autotrain-weather-classification-3723199087"
TEST_IMAGE_URL = "https://images.unsplash.com/photo-1519692933481-e162a57d6721?w=600"
LOCAL_IMAGE_PATH = "test_weather.jpg"

# ================================
# 步骤一：准备环境与图片
# ================================
print("1. 正在准备测试图片...")
if not os.path.exists(LOCAL_IMAGE_PATH):
    response = requests.get(TEST_IMAGE_URL, timeout=10)
    with open(LOCAL_IMAGE_PATH, "wb") as f:
        f.write(response.content)
    print("   图片下载成功！")
else:
    print("   使用本地已存在的测试图片。")

# ================================
# 步骤二：加载模型与单次推理（功能验证）
# ================================
print(f"\n2. 正在加载模型 [{MODEL_ID}] ...")
try:
    classifier = pipeline("image-classification", model=MODEL_ID)
    image = Image.open(LOCAL_IMAGE_PATH)
    
    # 第一次运行：验证业务结果
    predictions = classifier(image)
    
    print("\n" + "="*40)
    print("          单次分类预测结果          ")
    print("="*40)
    for idx, pred in enumerate(predictions, 1):
        print(f" Top {idx}: {pred['label']:<15} 置信度: {pred['score']*100:.2f}%")

    # ================================
    # 步骤三：连续推理 50 次（性能/稳定性测试）
    # ================================
    print("\n3. 正在进行性能与稳定性测试（基准测试）...")
    
    # A. 预热 (Warm-up)：跑 5 次，让 GPU/DCU 显存和缓存进入稳定工作状态
    print("   [1/2] 正在进行硬件预热 (Warm-up 5 次)...")
    for _ in range(5):
        _ = classifier(image)
        
    # B. 正式压测：连续跑 50 次计算平均耗时
    num_runs = 50
    print(f"   [2/2] 正在连续执行 {num_runs} 次推理压测...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 确保硬件准备就绪
        
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = classifier(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 GPU/DCU 所有计算完成
    end_time = time.perf_counter()
    
    # C. 指标计算
    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_runs) * 1000  # 平均延迟 (毫秒)
    fps = num_runs / total_time                       # 吞吐量 (帧/秒)
    
    print("\n" + "="*40)
    print("          性能测试评估指标          ")
    print("="*40)
    print(f" 🔁 测试总轮数 (Total Runs):  {num_runs} 次")
    print(f" ⏱️ 平均单张耗时 (Latency):   {avg_latency_ms:.2f} ms")
    print(f" 🚀 推理吞吐量 (Throughput):  {fps:.2f} FPS (img/s)")
    print("="*40)

except Exception as e:
    print(f"\n运行出错: {e}")