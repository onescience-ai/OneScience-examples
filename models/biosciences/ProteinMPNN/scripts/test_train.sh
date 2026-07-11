#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ONESCIENCE_ROOT="${ONESCIENCE_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}/model:${ONESCIENCE_ROOT}/src:${PYTHONPATH:-}"

PATH_FOR_TRAINING_DATA="${PROJECT_ROOT}/data/pdb_2021aug02_sample"

python "${PROJECT_ROOT}/scripts/training.py" \
           --path_for_outputs "${PROJECT_ROOT}/outputs/train/exp_020" \
           --path_for_training_data "$PATH_FOR_TRAINING_DATA" \
           --num_examples_per_epoch 1000 \
           --save_model_every_n_epochs 50
