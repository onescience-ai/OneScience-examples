#!/usr/bin/env bash
# =============================================================================
# DGMR_SolRad 一键运行脚本
# 环境: SCNET 超算  flagos_earth_onecode 镜像 / Hygon DCU K100AI
#
# 用法(在容器终端):
#     bash run.sh                              # 默认 DGMR_SO + 202504131100
#     bash run.sh DGMR_SO 202504161200         # 指定 模型类型 + 基准时刻
#     bash run.sh Generator_only 202507151200
#     USE_FLAGGEMS=0 bash run.sh               # 不启用 FlagGems,跑原生 torch 基线
#
# 前提: 已按 0_操作教程_START_HERE.md 把【完整仓库】(含 LFS 权重/数据 + model_architect
#       + 本包的 test.py)放在 /root/private_data/DGMR_SolRad 下。
# =============================================================================
set -e

WORKDIR="/root/private_data/DGMR_SolRad"
MODEL_TYPE="${1:-DGMR_SO}"          # DGMR_SO 或 Generator_only
BASETIME="${2:-202504131100}"      # 需与 sample_data 中存在的样本时间一致

echo "==================== DGMR_SolRad run.sh ===================="
echo " 工作目录 : ${WORKDIR}"
echo " 模型类型 : ${MODEL_TYPE}"
echo " 基准时刻 : ${BASETIME}"
echo " FlagGems : ${USE_FLAGGEMS:-1} (1=启用, 0=原生torch基线)"
echo "==========================================================="

cd "${WORKDIR}"

# 1) 关键文件检查(缺 LFS 大文件是最常见的失败原因)
NEED=( "test.py" "model_architect" \
       "model_weights/${MODEL_TYPE}/ft36/weights.ckpt" \
       "sample_data/sample_${BASETIME}.npz" )
for f in "${NEED[@]}"; do
  if [ ! -e "$f" ]; then
    echo "[缺失] $f"
    echo ">> 若缺的是 weights.ckpt / *.npz : 说明 Git LFS 大文件未就位,"
    echo "   请先在有网环境执行  bash download_repo.sh  (git lfs pull / hf download)。"
    echo ">> 若缺的是 test.py / model_architect : 把它们复制/下载到 ${WORKDIR} 下。"
    exit 1
  fi
done
# 额外提醒:LFS 指针文件通常只有几百字节,顺手看一眼大小
echo "[检查] 关键文件存在 ✓  (权重大小如下,应为 MB 级而非几百字节)"
du -sh "model_weights/${MODEL_TYPE}/ft36/weights.ckpt" "sample_data/sample_${BASETIME}.npz" 2>/dev/null || true

# 2) 依赖检查(镜像一般自带 torch/numpy;仅在缺失时提示安装)
python - <<'PY'
import importlib
miss = []
for mod, pip_name in [("numpy", "numpy==1.26.4"), ("torch", "torch"), ("einops", "einops==0.8.0")]:
    try:
        importlib.import_module(mod)
    except Exception:
        miss.append(pip_name)
if miss:
    print("[依赖] 缺失: " + ", ".join(miss))
    print("       如需安装(镜像通常禁外网,可用内网 pip 源):  pip install " + " ".join(miss))
else:
    print("[依赖] numpy / torch / einops 均已就绪 ✓")
PY

# 3) 运行,并把完整输出同时写入日志文件(日志用于粘贴进测试报告)
LOG="run_${BASETIME}_${MODEL_TYPE}.log"
echo "[运行] python test.py --model-type ${MODEL_TYPE} --basetime ${BASETIME}"
USE_FLAGGEMS="${USE_FLAGGEMS:-1}" \
  python test.py --model-type "${MODEL_TYPE}" --basetime "${BASETIME}" 2>&1 | tee "${LOG}"

echo "==========================================================="
echo "[完成] 预测输出 : pred_${BASETIME}_${MODEL_TYPE}.npy"
echo "[完成] 运行日志 : ${LOG}   <-- 把其中的关键行粘贴到测试报告"
echo "[完成] 算子日志 : flaggems_ops.log (若启用了 FlagGems)"
echo "==========================================================="
