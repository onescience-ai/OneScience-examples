#!/bin/bash
# NOAA ESD Coral Bleaching ViT 分类器 - 下载脚本
# 下载小文件和示例图片，大文件可选

mkdir -p /root/private_data/wyx/coral_bleaching_vit
cd /root/private_data/wyx/coral_bleaching_vit

echo "正在下载配置文件..."
wget -q https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/config.json
wget -q https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/preprocessor_config.json
wget -q https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/config.yaml
wget -q https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/.gitattributes

echo "正在下载示例图片..."
wget -q https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/00_example.png
#为对比明显，图片文件名称在test.py中改成01_example.png
wget -q https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/02_example.png

echo "配置文件和示例图片下载完成！"
echo ""
echo "是否下载模型权重？（343MB，可选）"
echo "下载 .pt 权重: wget https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/noaa-esd-coral-bleaching-vit-classifier-v1.pt"
echo "下载 .safetensors 权重: wget https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/model.safetensors"
echo "下载 ONNX 权重: wget https://huggingface.co/akridge/noaa-esd-coral-bleaching-vit-classifier-v1/resolve/main/noaa-esd-coral-bleaching-vit-classifier-v1.onnx"

echo ""
ls -lh
