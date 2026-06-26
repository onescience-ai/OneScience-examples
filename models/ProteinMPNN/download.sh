#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 11 个大文件
modelscope download --model OneScience/ProteinMPNN soluble_model_weights/v_48_020.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN soluble_model_weights/v_48_010.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN soluble_model_weights/v_48_002.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN soluble_model_weights/v_48_030.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN ca_model_weights/v_48_002.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN ca_model_weights/v_48_010.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN ca_model_weights/v_48_020.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN vanilla_model_weights/v_48_002.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN vanilla_model_weights/v_48_010.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN vanilla_model_weights/v_48_020.pt --local_dir ./
modelscope download --model OneScience/ProteinMPNN vanilla_model_weights/v_48_030.pt --local_dir ./
