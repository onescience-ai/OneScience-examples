# ===== FlagGems兼容补丁（必须放在最顶部）=====
import flag_gems
flag_gems.enable(
    unused=["batch_norm", "batch_norm_backward"],
    record=True,
    path="./gems_debug.log",
    once=True
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 独立ResNet结构
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x):
        residual = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class ForecastResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResBlock(1,16)
        self.block2 = ResBlock(16,32)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(32, 1)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        return self.head(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = ForecastResNet().to(device).eval()
    # 打印模型结构，和别人输出对齐
    print(model)

    batch_size = 2
    print("\n===== 开始运行测试预测 =====")
    loop_times = 50
    for i in range(loop_times):
        dummy_input = torch.randn(batch_size,1,224,224).to(device)
        with torch.no_grad():
            pred = model(dummy_input)
        if (i+1) % 10 == 0:
            print(f"已完成 {i+1}/{loop_times} 轮预测")

    print("\n✅ 全部模型前向传播执行完成！")
    print(f"单次输入shape {dummy_input.shape}, 输出shape {pred.shape}")
    print("📄 算子日志 gems_debug.log 已生成在当前目录！")

if __name__ == "__main__":
    main()