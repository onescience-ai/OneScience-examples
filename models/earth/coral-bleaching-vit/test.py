"""
NOAA ESD Coral Bleaching ViT 分类器 - 测试脚本
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import math

try:
    from safetensors.torch import load_file
except ImportError:
    print("请先安装 safetensors: pip install safetensors")
    raise


# ============ ViT 结构（键名完全匹配 transformers） ============

class ViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 注意：transformers 里叫 patch_embeddings.projection
        self.patch_embeddings = nn.Conv2d(
            config["num_channels"], 
            config["hidden_size"],
            kernel_size=config["patch_size"],
            stride=config["patch_size"]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        num_patches = (config["image_size"] // config["patch_size"]) ** 2
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embeddings(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, patch_embeds), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = config["hidden_size"] // config["num_attention_heads"]
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)
        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class ViTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
    
    def forward(self, hidden_states):
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.intermediate_act_fn = nn.GELU()
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.layernorm_after = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
    
    def forward(self, hidden_states):
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config["num_hidden_layers"])])
    
    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class ViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
    
    def forward(self, pixel_values):
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = self.layernorm(encoder_outputs)
        return sequence_output


class ViTForImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config["hidden_size"], config["num_labels"])
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        sequence_output = outputs[:, 0]
        logits = self.classifier(sequence_output)
        return logits


# ============ 加载和测试 ============

def get_config(config_path="config.json"):
    with open(config_path, "r") as f:
        hf_config = json.load(f)
    
    config = {
        "hidden_size": hf_config.get("hidden_size", 768),
        "num_hidden_layers": hf_config.get("num_hidden_layers", 12),
        "num_attention_heads": hf_config.get("num_attention_heads", 12),
        "intermediate_size": hf_config.get("intermediate_size", 3072),
        "hidden_dropout_prob": hf_config.get("hidden_dropout_prob", 0.0),
        "attention_probs_dropout_prob": hf_config.get("attention_probs_dropout_prob", 0.0),
        "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-12),
        "image_size": hf_config.get("image_size", 224),
        "patch_size": hf_config.get("patch_size", 16),
        "num_channels": hf_config.get("num_channels", 3),
        "num_labels": len(hf_config.get("id2label", {})),
    }
    return config, hf_config.get("id2label", {"0": "CORAL", "1": "CORAL_BL"})


def load_model(safetensors_path="model.safetensors", config_path="config.json"):
    config, id2label = get_config(config_path)
    model = ViTForImageClassification(config)
    state_dict = load_file(safetensors_path)
    
    # 键名映射：把 .projection. 替换掉，匹配 nn.Conv2d 的命名
    new_state_dict = {}
    for k, v in state_dict.items():
        if "patch_embeddings.projection." in k:
            k = k.replace("patch_embeddings.projection.", "patch_embeddings.")
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model, id2label


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def test_model():
    safetensors_path = "model.safetensors"
    config_path = "config.json"
    
    if not os.path.exists(safetensors_path):
        print(f"⚠️ 模型权重不存在: {safetensors_path}")
        return
    
    print("正在加载模型...")
    model, id2label = load_model(safetensors_path, config_path)
    print(f"✅ 模型加载成功！权重完全匹配")
    print(f"类别映射: {id2label}")
    
    transform = get_transforms()
    test_images = [ "01_example.png", "02_example.png"]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"⚠️ 图片不存在: {img_path}")
            continue
        
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            prediction = outputs.argmax(-1).item()
            confidence = probs[0][prediction].item()
        
        label = id2label.get(str(prediction), id2label.get(prediction, f"类别{prediction}"))
        print(f"{img_path}: {label} (置信度: {confidence:.4f})")


if __name__ == "__main__":
    test_model()
