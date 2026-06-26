#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 2 个大文件
modelscope download --model OneScience/evo2 checkpoints/evo2_nemo_7b/weights/__0_0.distcp --local_dir ./
modelscope download --model OneScience/evo2 checkpoints/evo2_nemo_7b/weights/__0_1.distcp --local_dir ./
