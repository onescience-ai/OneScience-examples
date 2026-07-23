#!/usr/bin/env bash
# ============================================================
# predictia/cerra_tas_vqvae 适配测试一键脚本
# 用法:
#   cd /root/private_data/Cerra_tas_vqvae
#   bash run.sh
#
# 可选环境变量:
#   USE_FLAGGEMS=0        关闭 FlagGems 算子加速
#   BLOCK_TENSORFLOW=0    不屏蔽 tensorflow(默认屏蔽,规避镜像内坏 .so)
#   SKIP_INSTALL=1        跳过依赖检查/安装
# ============================================================
set -e
cd "$(dirname "$0")"

# 让下游库不要去探测 TensorFlow / JAX(镜像内 libtensorflow_cc.so.2 有 undefined symbol)
export USE_TORCH=1
export USE_TF=0
export USE_JAX=0
export USE_FLAX=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

echo "==== [1/3] 检查/安装依赖 ===="
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1,跳过。"
else
    NEED_INSTALL=0
    python - <<'PY' || NEED_INSTALL=1
import importlib.util, sys
missing = [m for m in ("torch", "diffusers", "sklearn", "numpy")
           if importlib.util.find_spec(m) is None]
if missing:
    print("缺失: " + ", ".join(missing))
sys.exit(1 if missing else 0)
PY

    if [ "${NEED_INSTALL:-0}" = "1" ]; then
        echo "检测到缺失依赖,尝试 pip 安装(需外网)..."
        # 注意:不要安装/升级 torch —— 必须用镜像自带的 DCU 版
        pip install --no-cache-dir "diffusers>=0.21,<0.31" safetensors scikit-learn matplotlib || {
            echo "!! 依赖安装失败。若容器无外网,可先在有外网的机器上 pip download 后离线安装。"
            echo "   (即使 diffusers 装不上也不致命:test.py 会自动回退到 vqvae_torch.py 纯 torch 实现)"
        }
    else
        echo "依赖已就绪。"
    fi
fi

echo ""
echo "==== [2/3] 检查模型权重 ===="
if [ ! -f "diffusion_pytorch_model.bin" ] && [ ! -f "diffusion_pytorch_model.safetensors" ]; then
    echo "!! 未发现权重文件 diffusion_pytorch_model.bin"
    echo "   请先把该文件放到当前目录(见 README.md)。"
    echo "   若容器有外网,也可执行:  python download_weights.py"
    exit 1
fi
echo "权重文件已就绪。"

echo ""
echo "==== [3/3] 运行适配测试 ===="
export USE_FLAGGEMS=${USE_FLAGGEMS:-1}
export BLOCK_TENSORFLOW=${BLOCK_TENSORFLOW:-1}
set +e
python test.py --save_fig "$@"
RC=$?
set -e
echo ""
if [ "$RC" = "0" ]; then
    echo "==== 完成(PASS)。结果见 result.log / result.json ===="
else
    echo "==== 运行未通过(退出码 $RC)。请把 result.log 完整发回排查。 ===="
    echo "     可先跑诊断脚本:  python diagnose.py     (输出 diagnose.log)"
fi
exit $RC
