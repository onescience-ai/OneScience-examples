#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. 下载 DeepR（如已存在则跳过）
[ -d "DeepR" ] || git clone https://github.com/ECMWFCode4Earth/DeepR.git

# 2. 下载模型（如已存在则跳过）
MODEL_DIR="europe_reanalysis_downscaler_convswin2sr"
[ -d "$MODEL_DIR" ] || git clone https://huggingface.co/predictia/europe_reanalysis_downscaler_convswin2sr

# 3. 进入模型目录并运行测试
cd "$MODEL_DIR"
if [ -f "test_model.py" ]; then
    python test_model.py
else
    echo "❌ 未找到 test_model.py，请确认它在模型目录中"
    exit 1
fi