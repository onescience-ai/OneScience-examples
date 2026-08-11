# test_with_deepr.py
# 这个脚本位于模型文件夹内，会自动引用 DeepR 项目的代码

import sys
import os
import json
import torch
import numpy as np
from PIL import Image

# ---------- 添加 DeepR 项目路径 ----------
#deepr请从github中https://github.com/ECMWFCode4Earth/DeepR进行下载，然后更改为自己的文件夹的路径，建议与europe_reanalysis_downscaler_convbaseline位于同一级目录下
deepr_path = "../DeepR"
if deepr_path not in sys.path:
    sys.path.insert(0, deepr_path)

# ---------- 导入 DeepR 的模型加载函数 ----------
from deepr.model import models

# ---------- 当前模型文件夹路径 ----------
#----------请用户根据实际情况修改路径----------
 # 脚本本身就在模型文件夹内，所以当前目录就是模型目录
model_dir = "." 
# ---------- 读取 config.json ----------
config_path = os.path.join(model_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
print("✅ 配置文件读取成功")
print("   model_type:", config.get("model_type"))
print("   input_shape:", config.get("input_shape"))
print("   upscale:", config.get("upscale"))

# ---------- 加载模型 ----------
print("\n⏳ 正在使用 DeepR 加载 ConvBaseline 模型...")
try:
    model = models.load_trained_model(
        class_name="ConvBaseline",
        model_dir=model_dir
    )
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ---------- 准备输入数据 ----------
# 从 config 获取输入尺寸（默认 44x60）
input_height, input_width = config.get("input_shape", [44, 60])
print(f"\n⏳ 生成模拟数据，尺寸 {input_height}x{input_width} ...")

# 生成随机灰度图（0~255）
dummy_data = np.random.randint(0, 255, size=(input_height, input_width), dtype=np.uint8)
input_image = Image.fromarray(dummy_data, mode='L')

# 转换为 PyTorch 张量
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),          # 转为 [0,1] 范围
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1,1]
])
input_tensor = transform(input_image).unsqueeze(0)  # 添加 batch 维度

# ---------- 推理 ----------
print("⏳ 正在推理...")
with torch.no_grad():
    try:
        # 尝试用 "pixel_values" 作为输入参数名
        outputs = model(pixel_values=input_tensor)
    except TypeError:
        try:
            # 或者直接传入张量
            outputs = model(input_tensor)
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            exit(1)

# ---------- 提取输出 ----------
print(f"输出类型: {type(outputs)}")

# 如果是元组，尝试取第一个元素
if isinstance(outputs, tuple):
    print(f"元组长度: {len(outputs)}")
    for i, item in enumerate(outputs):
        print(f"  元素 {i}: {type(item)}, 形状: {item.shape if hasattr(item, 'shape') else 'N/A'}")
    # 假设第一个元素是输出张量
    output_tensor = outputs[0]
elif hasattr(outputs, 'reconstruction'):
    output_tensor = outputs.reconstruction
elif hasattr(outputs, 'logits'):
    output_tensor = outputs.logits
elif isinstance(outputs, torch.Tensor):
    output_tensor = outputs
else:
    print(f"❌ 未知输出类型: {type(outputs)}")
    exit(1)

# ---------- 保存结果 ----------
output_np = output_tensor.squeeze().cpu().numpy()
# 归一化到 0~255
min_val = np.min(output_np)
max_val = np.max(output_np)
if max_val - min_val > 1e-8:
    output_np = (output_np - min_val) / (max_val - min_val) * 255
else:
    output_np = np.zeros_like(output_np)  # 防止全黑或全白
output_np = output_np.astype(np.uint8)
out_img = Image.fromarray(output_np, mode='L')

output_path = os.path.join(model_dir, "output_final_success.png")
out_img.save(output_path)

print(f"\n✅ 推理成功！结果已保存至：{output_path}")