import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import Swin2SRConfig
from deepr.model.conv_swin2sr import ConvSwin2SR

MODEL_PATH = "/root/private_data/lyc/convswin2sr_mediterranean"

# 1. 加载 config.json 并过滤掉 Hugging Face 元数据键
with open(f"{MODEL_PATH}/config.json", "r") as f:
    config_dict = json.load(f)
for key in ['architectures', 'model_type', 'transformers_version']:
    config_dict.pop(key, None)

# 2. 创建 Swin2SRConfig 对象
config = Swin2SRConfig(**config_dict)

# 3. 为 config 添加 swin2sr_kwargs 方法（返回 config 自身）
def swin2sr_kwargs(self):
    return self
config.swin2sr_kwargs = swin2sr_kwargs.__get__(config, Swin2SRConfig)

# 4. 加载标准化参数
with open(f"{MODEL_PATH}/training_scale.json", "r") as f:
    scale_params = json.load(f)

# 5. 实例化模型
model = ConvSwin2SR(config)

# 6. 手动加载权重
state_dict = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location="cpu", weights_only=False)
model.load_state_dict(state_dict)
model.eval()
print("✅ 模型加载成功！")

# 7. 模拟输入
month = 1
dummy_lr_data = np.random.randn(44, 60) * 5 + 280
mean = scale_params['features']['average'][month-1]
std = scale_params['features']['standard_deviation'][month-1]
normalized = (dummy_lr_data - mean) / std
input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)

# 8. 推理（处理返回的元组）
with torch.no_grad():
    output = model(input_tensor)
    if isinstance(output, tuple):
        output_tensor = output[0]   # 只取预测结果
    else:
        output_tensor = output

# 9. 反标准化
out_mean = scale_params['label']['average'][month-1]
out_std = scale_params['label']['standard_deviation'][month-1]
sr_data = output_tensor.squeeze().cpu().numpy() * out_std + out_mean

print(f"✅ 降尺度成功：输入 (44,60) -> 输出 {sr_data.shape}")

# 10. 保存图片
plt.figure(figsize=(10,8))
plt.imshow(sr_data, cmap='RdBu_r', origin='lower')
plt.colorbar(label='Temperature (K)')
plt.title("DeepR Downscaled t2m")
plt.savefig("downscaled_result.png")
print("🖼️  图片已保存为 downscaled_result.png")