import os
from huggingface_hub import snapshot_download

# 定义下载的目标目录
target_dir = "/root/group_data/SDU-Test/lmy/Weather-R1/dataset"

print(f"--- 正在从 Hugging Face 下载 WeatherQA 数据集 ---")
print(f"目标目录: {target_dir}")

try:
    # 使用 snapshot_download 下载指定文件
    # allow_patterns 确保只下载我们需要的数据文件和图片文件夹
    snapshot_download(
        repo_id="Marco711/WeatherQA",
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=["image/*", "all_split.json"],
        resume_download=True  # 支持断点续传
    )
    print("\n--- 下载成功完成！ ---")
    print(f"数据集已准备就绪，存放在: {target_dir}")
    print("现在您可以运行 ./run.sh 来启动模型适配测试了。")
except Exception as e:
    print(f"\n--- 下载过程中出现错误: {e} ---")
    print("提示: ")
    print("1. 请确保已安装 huggingface_hub 库 (pip install huggingface_hub)")
    print("2. 如果下载缓慢，可以尝试设置环境变量: $env:HF_ENDPOINT='https://hf-mirror.com' (Windows PowerShell)")
