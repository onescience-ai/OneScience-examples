#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 2 个大文件
modelscope download --model OneScience/MatRIS data/matris/pbesol_phonon_ref.csv --local_dir ./
modelscope download --model OneScience/MatRIS data/matris/pbe_phonon_ref.csv --local_dir ./
