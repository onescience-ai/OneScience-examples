#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 1 个大文件
modelscope download --model OneScience/AlphaFold3 inputs/7r6r_data.json --local_dir ./
