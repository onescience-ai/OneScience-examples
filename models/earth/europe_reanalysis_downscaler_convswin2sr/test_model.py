import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import Swin2SRConfig
from deepr.model.conv_swin2sr import ConvSwin2SR

MODEL_PATH = "."

# 1. 加载 config.json
with open(f"{MODEL_PATH}/config.json", "r") as f:
    config_dict = json.load(f)

# 移除 Hugging Face 保留字段
for key in ['architectures', 'model_type', 'transformers_version']:
    config_dict.pop(key, None)

# 2. 补充缺失属性（默认值为0，源自源码中的 getattr(..., 0)）
if 'num_high_res_covars' not in config_dict:
    config_dict['num_high_res_covars'] = 0
    print("⚠️ 自动添加 num_high_res_covars=0 (源码默认值)")
if 'num_low_res_covars' not in config_dict:
    config_dict['num_low_res_covars'] = 0
    print("⚠️ 自动添加 num_low_res_covars=0 (源码默认值)")

# 3. 创建配置对象
config = Swin2SRConfig(**config_dict)

def swin2sr_kwargs(self):
    return self
config.swin2sr_kwargs = swin2sr_kwargs.__get__(config, Swin2SRConfig)

# 4. 实例化模型
model = ConvSwin2SR(config)

# 5. 加载权重（strict=False 允许缺失键）
state_dict = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"⚠️ 缺失的键: {missing[:5]}... (如果这些是协变量相关，说明没问题)")
if unexpected:
    print(f"⚠️ 多余的键: {unexpected[:5]}...")

# --- 改为训练模式，启用梯度 ---
model.train()
print("✅ 模型加载完成，已切换到训练模式")

# 6. 生成输入：使用原尺寸 44x60
H, W = 44, 60
dummy_input = np.random.randn(H, W).astype(np.float32)
input_tensor = torch.from_numpy(dummy_input).unsqueeze(0).unsqueeze(0)
input_tensor.requires_grad_(True)   # 启用梯度
print(f"📥 输入形状: {input_tensor.shape}")

# ---------- 打印模型参数类型和输入类型/形状 ----------
print("model_dtypes:", sorted({str(p.dtype) for p in model.parameters()}), "input_dtype:", input_tensor.dtype)
print("input_tensor_shape:", input_tensor.shape)

# 7. 前向传播（去掉 no_grad）
output = model(input_tensor)
if isinstance(output, tuple):
    output_tensor = output[0]
else:
    output_tensor = output
print(f"📤 输出形状: {output_tensor.shape}")

# 8. 构造损失并反向传播
target = torch.zeros_like(output_tensor)
loss = torch.nn.functional.mse_loss(output_tensor, target)
loss.backward()
print("✅ 前向+反向传播完成")

# 9. 可视化
sr_data = output_tensor.detach().squeeze().cpu().numpy()
plt.figure(figsize=(10,8))
plt.imshow(sr_data, cmap='RdBu_r', origin='lower')
plt.colorbar(label='Arbitrary units')
plt.title("DeepR 测试输出 (输入 44x60)")
plt.savefig("test_output.png")
print("🖼️ 图片已保存为 test_output.png")
