#!/usr/bin/env python
# coding: utf-8

# In[33]:


import sys
import torch
from huggingface_hub import snapshot_download
from tornado_predictor import TornadoSuperPredictor
import time

def load_tornet_oracle(repo="Wonder-Griffin/TorNet-Oracle"):

    print("正在从 Hugging Face 获取模型仓库...")
    local = snapshot_download(repo)
    sys.path.insert(0, local)
    
    model = TornadoSuperPredictor(in_channels=9).eval()
    state = torch.load(f"{local}/pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    
    sys.path.pop(0)
    return model

if __name__ == "__main__":
    start = time.time()
    model = load_tornet_oracle()
    
    print("\n正在执行前向推理...")
    with torch.no_grad():
        radar_x = torch.randn(2, 9, 256, 256) 
        
        atmo = {
            "cape":        torch.randn(2, 1),       
            "wind_shear":  torch.randn(2, 4), 
            "helicity":    torch.randn(2, 2),   
            "temperature": torch.randn(2, 3), 
            "dewpoint":    torch.randn(2, 2),   
            "pressure":    torch.randn(2, 1)
        }

        out = model(radar_x=radar_x, atmo=atmo)
        params = sum(p.numel() for p in model.parameters())

        end = time.time()
        
        print("龙卷风发生概率:", torch.sigmoid(out.tornado_probability).squeeze().tolist())
        print("EF 级数预测 (概率最大项):", out.ef_scale_probs.argmax(dim=-1).tolist())
        print("位置偏移量形状:", out.location_offset.shape)
        print("发生时间预测形状:", out.timing_predictions.shape)
        print("推理耗时:", (end-start)*1000, "ms")
        print(params)
