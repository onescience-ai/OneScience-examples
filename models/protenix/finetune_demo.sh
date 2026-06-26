#!/bin/bash
set -euo pipefail
if [ -f ../../../env.sh ]; then
  source ../../../env.sh
fi
checkpoint_path=${LOAD_CHECKPOINT_PATH:-"./checkpoints/model_v0.5.0.pt"}

export PYTHONPATH=$(pwd):$PYTHONPATH
export DATA_ROOT_DIR=${DATA_ROOT_DIR:-"../bio_protenix_dataset"}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

python3 ./runner/train.py \
--run_name protenix_finetune \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb false \
--diffusion_batch_size 48 \
--eval_interval 400 \
--log_interval 50 \
--checkpoint_interval 400 \
--ema_decay 0.999 \
--train_crop_size 384 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
--data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list ft_datasets/finetune_subset.txt \
--data.test_sets recentPDB_1536_sample384_0925,posebusters_0925
