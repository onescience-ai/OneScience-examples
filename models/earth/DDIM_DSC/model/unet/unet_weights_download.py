#!/usr/bin/env python
# coding: utf-8

# In[3]:


from huggingface_hub import hf_hub_download
import os

# 直接下载到当前目录，不创建任何子文件夹
file_path = hf_hub_download(
    repo_id="lschmidt/ddim-dsc",
    filename="diffusion_pytorch_model.safetensors",
    subfolder="unet",
    local_dir="./",           # 直接下载到当前目录
    local_dir_use_symlinks=False,
    resume_download=True,
)

# 如果文件被下载到 ./unet/ 下，移动它到当前目录并删除 unet 目录
import shutil

# 检查文件是否在 unet 子目录中
expected_path = "./unet/diffusion_pytorch_model.safetensors"
if os.path.exists(expected_path) and expected_path != file_path:
    # 移动到当前目录
    shutil.move(expected_path, "./diffusion_pytorch_model.safetensors")
    # 删除空的 unet 目录
    os.rmdir("./unet")
    file_path = "./diffusion_pytorch_model.safetensors"

print(f"✅ 文件已下载到: {file_path}")
print(f"📊 大小: {os.path.getsize(file_path) / (1024**2):.2f} MB")


# In[ ]:




