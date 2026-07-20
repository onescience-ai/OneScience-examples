"""
飓风图像分类 - 测试推理脚本
使用训练好的模型进行预测
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def load_model(checkpoint_path="./best_model.pth", num_classes=2):
    """加载训练好的模型"""
    model = models.vit_b_16(weights=None)
    model.heads = nn.Sequential(nn.Linear(model.hidden_dim, num_classes))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

def predict(image_path, model=None, class_names=None):
    """对单张图片进行预测"""
    if model is None:
        model = load_model()
    
    if class_names is None:
        class_names = ['flooded or damaged buildings', 'undamaged buildings']
    
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    
    return {
        "prediction": class_names[pred],
        "confidence": f"{confidence:.4f}",
        "class_id": pred
    }

# ========== 测试推理 ==========
if __name__ == "__main__":
    # 检查当前目录下有没有图片可以测试
    current_dir = os.listdir(".")
    image_files = [f for f in current_dir if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if image_files:
        test_image = image_files[0]
        print(f"使用测试图片: {test_image}")
        result = predict(test_image)
        print(f"预测结果: {result}")
    else:
        print("当前目录没有测试图片，请提供图片路径")
        print("用法: result = predict('your_image.jpg')")
        
        # 演示用随机张量测试模型是否能正常推理
        print("\n使用随机张量测试模型...")
        model = load_model()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            pred = output.argmax(dim=1).item()
        print(f"随机输入测试通过，输出形状: {output.shape}, 预测类别: {pred}")
        