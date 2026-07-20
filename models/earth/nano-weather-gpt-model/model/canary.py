#!/usr/bin/env python3
"""Test nanoWeatherGPT — run: python3 canary.py"""
import torch
import pickle
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT, GPTConfig

# Load model
ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
meta = pickle.load(open("meta.pkl", "rb"))
stoi, itos = meta["stoi"], meta["itos"]

model = GPT(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generate
prompt = "Today the weather"
ctx = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=200)

print("".join(itos[i] for i in out[0].tolist()))
