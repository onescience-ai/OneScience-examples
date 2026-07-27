#!/bin/bash
# RainMap Nowcasting 模型 - 下载脚本
# 下载测试所需的小文件

mkdir -p /root/private_data/wyx/rainmap-nowcasting
cd /root/private_data/wyx/rainmap-nowcasting

echo "正在下载配置文件..."
wget -q https://huggingface.co/TechieMoon/rainmap-nowcasting/resolve/main/model_config.json

echo "正在下载 demo 权重..."
wget -q https://huggingface.co/TechieMoon/rainmap-nowcasting/resolve/main/rainmap-nowcasting-demo.safetensors

echo "下载完成！"
ls -lh
