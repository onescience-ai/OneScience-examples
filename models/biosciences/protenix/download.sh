#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 3 个大文件
modelscope download --model OneScience/protenix infer_datasets/7pzb/msa/1/pairing.a3m --local_dir ./
modelscope download --model OneScience/protenix infer_datasets/7pzb/msa/1/non_pairing.a3m --local_dir ./
modelscope download --model OneScience/protenix checkpoints/model_v0.5.0.pt --local_dir ./
