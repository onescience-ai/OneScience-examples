#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 3 个大文件
modelscope download --model OneScience/alphagenome checkpoints/alphagenome-all-folds/ocdbt.process_0/d/6b08a83c9d6a2ade925dbcdd91299ce2 --local_dir ./
modelscope download --model OneScience/alphagenome checkpoints/alphagenome-all-folds/ocdbt.process_0/d/48900811ab32b5ad3873d40e96d63846 --local_dir ./
modelscope download --model OneScience/alphagenome checkpoints/alphagenome-all-folds/ocdbt.process_0/d/bda173d4f00d3afcef0a043614df05a6 --local_dir ./
