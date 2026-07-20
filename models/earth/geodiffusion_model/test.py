import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["DIFFUSERS_USE_FLAX"] = "0"

import torch
# 强制禁用所有 SDPA 优化后端
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)  # 强制使用数学回退

from diffusers import StableDiffusionPipeline

model_path = "/root/private_data/KaiChen1998/geodiffusion_model"
print("Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager"   # 明确使用 eager 实现
).to("cuda")
print("Pipeline loaded!")

prompt = "a car driving on a rainy night, realistic"
print("Generating image...")
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("test_output.png")
print("✅ 生成成功！图片已保存为 test_output.png")