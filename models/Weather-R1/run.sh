#!/bin/bash

# ==============================================================================
# Weather-R1 模型适配测试脚本 (集成 flag_gems)
# ==============================================================================

# 1. 基础依赖安装与环境清理
  echo "--- 正在清理冲突依赖并安装必要包 ---"
  # 卸载可能导致环境崩溃或冲突的无关框架
  pip uninstall -y tensorflow jax jaxlib vllm
  pip install flag_gems transformers qwen-vl-utils tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 设置项目根目录到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. 参数配置
# 注意：请确保已从 Hugging Face 下载模型权重，或指定正确的模型 ID
MODEL_PATH=${1:-"Marco711/Weather-R1"}
DATA_FILE=${2:-"path/to/your/test_data.json"}
IMAGE_FOLDER=${3:-"path/to/your/images"}

# 根据模型路径动态生成输出文件名
MODEL_NAME=$(basename "$MODEL_PATH")
mkdir -p results
OUTPUT_FILE="results/adaptation_${MODEL_NAME}.jsonl"

# 4. 运行适配测试
echo "--- 正在启动模型适配测试 (使用 flag_gems 优化) ---"
echo "模型路径: $MODEL_PATH"

if [ ! -f "$DATA_FILE" ]; then
    echo "错误: 未找到测试数据文件 $DATA_FILE"
    echo "请通过参数运行: ./run.sh [模型路径] [数据文件路径] [图片文件夹路径]"
    exit 1
fi

python -m src.eval.gen_ans_multi_gpu \
    --data-file "$DATA_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$OUTPUT_FILE" \
    --model-name "$MODEL_PATH" \
    --temperature 0 \
    --language "cn" \
    --gpu-id 0 \
    --gpu_num 1

echo "--- 测试完成 ---"
echo "1. 运行日志请查看标准输出"
echo "2. flag_gems 算子记录请查看: ./gems_debug.log"
echo "3. 模型输出结果请查看: $OUTPUT_FILE"
