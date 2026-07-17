#!/usr/bin/env bash
set -eo pipefail

source "${HOME}/env.sh"
cd /public/home/liuyushuang/code/onecode_new_model/paper_2605_31559_original_airfrans

python scripts/train.py --config config/config.yaml 2>&1 | tee -a weight/train_reynolds_e02r1n03.log
