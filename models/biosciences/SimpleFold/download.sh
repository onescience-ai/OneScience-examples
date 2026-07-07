#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 10 个大文件
modelscope download --model OneScience/SimpleFold checkpoints/ccd.pkl --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/simplefold_100M.ckpt --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/plddt_module_1.6B.ckpt --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/simplefold_360M.ckpt --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/simplefold_700M.ckpt --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/boltz1_conf.ckpt --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/simplefold_1.6B.ckpt --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/plddt.ckpt --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/simplefold_1.1B.ckpt --local_dir ./
modelscope download --model OneScience/SimpleFold checkpoints/simplefold_3B.ckpt --local_dir ./
