#!/bin/bash
set -euo pipefail
# OneScience 标准包推理入口。默认使用本仓库内权重和相邻数据集目录。
if [ -f ../../../env.sh ]; then
  source ../../../env.sh
fi
if [ -n "${ROCM_PATH:-}" ] && [ -f "${ROCM_PATH}/cuda/env.sh" ]; then
  source "${ROCM_PATH}/cuda/env.sh"
fi

N_sample=${N_sample:-5}
N_step=${N_step:-200}
N_cycle=${N_cycle:-10}
seed=${seed:-101}
use_deepspeed_evo_attention=${USE_DEEPSPEED_EVO_ATTENTION:-false}
input_json_path=${INPUT_JSON_PATH:-"./infer_datasets/7r6r.json"}
load_checkpoint_path=${LOAD_CHECKPOINT_PATH:-"./checkpoints/model_v0.5.0.pt"}
dump_dir=${DUMP_DIR:-"./output_unified"}

export PYTHONPATH=$(pwd):$PYTHONPATH
export DATA_ROOT_DIR=${DATA_ROOT_DIR:-"../bio_protenix_dataset"}

export USE_DEEPSPEED_EVO_ATTENTION=${use_deepspeed_evo_attention}

echo "======================================"
echo "Unified Interface Inference"
echo "======================================"
echo "Input: $input_json_path"
echo "Output: $dump_dir"
echo "Checkpoint: $load_checkpoint_path"
echo ""

python3 runner/inference_unified.py \
    --seeds ${seed} \
    --dtype bf16 \
    --num_workers 8 \
    --load_checkpoint_path ${load_checkpoint_path} \
    --dump_dir ${dump_dir} \
    --input_json_path ${input_json_path} \
    --model.N_cycle ${N_cycle} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --use_msa true

echo ""
echo "Inference completed! Results saved to: $dump_dir"
