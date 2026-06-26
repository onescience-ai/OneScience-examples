#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 3 个大文件
modelscope download --model OneScience/OpenFold monomer/alignments/6KWC_1/mgnify_hits.sto --local_dir ./
modelscope download --model OneScience/OpenFold monomer/alignments/6KWC_1/uniref90_hits.sto --local_dir ./
modelscope download --model OneScience/OpenFold params/finetuning_ptm_2.pt --local_dir ./
