"""
飓风图像分类模型 - PyTorch torchvision 实现
基于 ViT-Base/16 微调
数据集: jonathan-roberts1/Satellite-Images-of-Hurricane-Damage
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
import os

print(f"PyTorch: {torch.__version__}")

# ========== 1. 加载数据集 ==========
dataset = load_dataset("jonathan-roberts1/Satellite-Images-of-Hurricane-Damage")

print(f"可用的数据划分: {list(dataset.keys())}")

num_classes = len(dataset["train"].features["label"].names)
class_names = dataset["train"].features["label"].names
print(f"类别: {class_names}")
print(f"原始训练集: {len(dataset['train'])}")

# 划分: 80% train, 10% val, 10% test
dataset_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
val_test_split = dataset_split["test"].train_test_split(test_size=0.5, seed=42)

train_data = dataset_split["train"]
val_data = val_test_split["train"]
test_data = val_test_split["test"]

print(f"训练集: {len(train_data)}")
print(f"验证集: {len(val_data)}")
print(f"测试集: {len(test_data)}")

# ========== 2. 图像预处理 ==========
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ========== 3. 数据转换函数 ==========
def transform_dataset(examples):
    images = [preprocess(img.convert("RGB")) for img in examples["image"]]
    labels = examples["label"]
    return {"pixel_values": images, "labels": labels}

# 应用转换
train_data = train_data.with_transform(transform_dataset)
val_data = val_data.with_transform(transform_dataset)
test_data = test_data.with_transform(transform_dataset)

# ========== 4. 创建DataLoader ==========
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=8, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=8, collate_fn=collate_fn)

# ========== 5. 加载模型 ==========
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# 修改分类头
model.heads = nn.Sequential(
    nn.Linear(model.hidden_dim, num_classes)
)

# CPU训练
device = torch.device("cpu")
model = model.to(device)

print(f"模型加载成功！参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ========== 6. 训练设置 ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-08)

# 线性学习率衰减
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=1.0, 
    end_factor=0.0, 
    total_iters=4
)

# ========== 7. 训练循环 ==========
num_epochs = 4
best_val_acc = 0.0

for epoch in range(num_epochs):
    # 训练
    model.train()
    total_loss = 0
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # 保存最佳模型
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), "./best_model.pth")
        print(f"  -> 保存最佳模型 (Val Acc: {val_accuracy:.4f})")

# ========== 8. 最终测试集评估 ==========
print("\n加载最佳模型进行测试...")
model.load_state_dict(torch.load("./best_model.pth"))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_labels, all_preds)
print(f"\n最终测试集准确率: {test_accuracy:.4f}")

# ========== 9. 保存完整模型 ==========
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'class_names': class_names,
    'preprocess': preprocess,
}, "./hurricane_model_complete.pth")
print("模型已保存: ./hurricane_model_complete.pth")
