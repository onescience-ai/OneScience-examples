import torch
import torch.nn as nn
import flag_gems

# 启用 FlagGems，记录日志
flag_gems.enable(
    unused=[],  # 本次不禁用任何算子
    record=True,
    path="./flaggems_test.log",
    once=True   # 每个算子只记录一次
)

print("🔍 FlagGems 已启用")
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())

# 创建一个简单的神经网络模型（包含常见算子）
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(50)  # 这个会触发 batch_norm，但我们已禁用列表为空，所以会被 FlagGems 接管

    def forward(self, x):
        x = self.fc1(x)          # addmm 或 mm
        x = self.bn(x)           # batch_norm
        x = self.relu(x)         # relu
        x = self.fc2(x)          # addmm
        return x

model = SimpleModel().cuda()
model.eval()

# 生成随机输入
x = torch.randn(32, 100).cuda()

# 执行推理
with torch.no_grad():
    y = model(x)
    torch.cuda.synchronize()   # 确保计算完成

print("✅ 推理完成，输出形状:", y.shape)
print("✅ 测试结束，请查看 flaggems_test.log")
