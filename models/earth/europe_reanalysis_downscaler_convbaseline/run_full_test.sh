#!/bin/bash

# 获取脚本所在目录（即父目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== 当前工作目录: $SCRIPT_DIR ==="

# ---------- 1. 下载 DeepR ----------
DEEPR_DIR="DeepR"
if [ -d "$DEEPR_DIR" ]; then
    echo "✅ DeepR 目录已存在，跳过下载"
else
    echo "⏳ 正在克隆 DeepR 项目 (https://github.com/ECMWFCode4Earth/DeepR) ..."
    git clone https://github.com/ECMWFCode4Earth/DeepR.git
    if [ $? -ne 0 ]; then
        echo "❌ DeepR 下载失败，请检查网络或 git 命令"
        exit 1
    fi
    echo "✅ DeepR 下载完成"
fi

# ---------- 2. 下载模型 ----------
MODEL_DIR="europe_reanalysis_downscaler_convbaseline"
if [ -d "$MODEL_DIR" ]; then
    echo "✅ 模型目录已存在，跳过下载"
else
    echo "⏳ 正在克隆模型仓库 (https://huggingface.co/predictia/europe_reanalysis_downscaler_convbaseline) ..."
    # Hugging Face 仓库使用 git-lfs，确保已安装
    git clone https://huggingface.co/predictia/europe_reanalysis_downscaler_convbaseline
    if [ $? -ne 0 ]; then
        echo "❌ 模型下载失败，请检查网络或 git 命令"
        exit 1
    fi
    echo "✅ 模型下载完成"
fi

# ---------- 3. 进入模型目录并运行测试 ----------
echo "⏳ 进入模型目录并执行 test_with_deepr.py ..."
cd "$MODEL_DIR"

# 检查测试脚本是否存在
if [ ! -f "test_with_deepr.py" ]; then
    echo "❌ 未找到 test_with_deepr.py，请确保它已放在模型目录内"
    exit 1
fi

# 运行测试（如果你的 Python 命令是 python3，请改为 python3）
python test_with_deepr.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "✅ 所有任务成功完成！输出图片已保存在模型目录内。"
else
    echo "❌ 测试脚本执行失败，请查看上方错误信息。"
    exit 1
fi