import torch
import pickle
import sys

# 固定本地仓库路径
path = "/root/private_data/csy/nano-weather-gpt-model"
# 将路径加入模块搜索，解决 from model 导入报错
sys.path.insert(0, path)
from model import GPT

ckpt = torch.load(f"{path}/model.pt", map_location="cpu", weights_only=False)
meta = pickle.load(open(f"{path}/meta.pkl", "rb"))
stoi, itos = meta["stoi"], meta["itos"]

model = GPT(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompt = "Today the weather is"
ctx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=200)
print("".join(itos[i] for i in out[0].tolist()))