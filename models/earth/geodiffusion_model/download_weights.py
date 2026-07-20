#!/usr/bin/env python3
"""
GeoDiffusion 模型权重下载脚本
从 Hugging Face 下载被删除的权重文件
"""

import os
import sys
from huggingface_hub import snapshot_download

def download_weights():
    print("正在下载 GeoDiffusion 模型权重...")
    print("目标路径:", os.getcwd())
    
    # 使用镜像站加速（如果网络不通，可改为官方源）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        snapshot_download(
            repo_id="KaiChen1998/geodiffusion-nuimages-time-weather-512x512",
            local_dir="./",
            local_dir_use_symlinks=False,
            ignore_patterns=["*.safetensors"],  # 不下载 safetensors（只用 bin）
        )
        print("✅ 所有权重文件下载完成！")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请检查网络连接，或手动从以下地址下载：")
        print("https://huggingface.co/KaiChen1998/geodiffusion-nuimages-time-weather-512x512")
        sys.exit(1)

if __name__ == "__main__":
    download_weights()