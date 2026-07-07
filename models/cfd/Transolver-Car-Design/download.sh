#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 1 个大文件
modelscope download --model OneScience/Transolver-Car-Design checkpoints/ShapeNetCar/Transolver_plus.pth --local_dir ./
