"""
RainMap Nowcasting - 测试脚本
基于 RainMapUNet 架构的降水临近预报模型
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file
import json
import os
import time


# ============ UNet 结构 ============

class ConvBlock(nn.Module):
    """Conv + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RainMapUNet(nn.Module):
    def __init__(self, input_frames=6, target_frames=6, base_channels=8):
        super().__init__()

        # 编码器
        self.enc1 = ConvBlock(input_frames, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # 上采样
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)

        # 解码器
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # 输出
        self.out = nn.Conv2d(base_channels, target_frames, 1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # 瓶颈
        b = self.bottleneck(self.pool(e3))

        # 解码 + skip
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ============ 加载和测试 ============

def load_model(config_path="model_config.json", weights_path="rainmap-nowcasting-demo.safetensors"):
    with open(config_path, "r") as f:
        config = json.load(f)

    model = RainMapUNet(
        input_frames=config["input_frames"],
        target_frames=config["target_frames"],
        base_channels=config["base_channels"]
    )

    state_dict = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print("   缺失键 ({}个): {}".format(len(missing), missing[:3]))
    if unexpected:
        print("   多余键 ({}个): {}".format(len(unexpected), unexpected[:3]))

    model.eval()
    return model, config


def test_random_input(model, config):
    """随机输入测试"""
    dummy = torch.randn(1, config["input_frames"], *config["image_size"])

    # 预热
    with torch.no_grad():
        _ = model(dummy)

    # 正式推理计时
    start = time.time()
    with torch.no_grad():
        output = model(dummy)
    elapsed = time.time() - start

    print("随机输入测试通过")
    print("   输入形状:  {}".format(dummy.shape))
    print("   输出形状:  {}".format(output.shape))
    print("   输出范围:  [{:.4f}, {:.4f}]".format(output.min(), output.max()))
    print("   输出均值:  {:.4f}".format(output.mean()))
    print("   推理耗时:  {:.4f}s".format(elapsed))

    return output


def test_model():
    config_path = "model_config.json"
    weights_path = "rainmap-nowcasting-demo.safetensors"

    if not os.path.exists(weights_path):
        print("权重文件不存在: {}".format(weights_path))
        print("请先运行 download.sh 下载权重")
        return

    print("=" * 50)
    print("RainMap Nowcasting 模型测试")
    print("=" * 50)

    print()
    print("正在加载模型...")
    model, config = load_model(config_path, weights_path)
    print("模型加载成功！")
    print("   类型: {}".format(config["model_type"]))
    print("   输入帧: {}".format(config["input_frames"]))
    print("   输出帧: {}".format(config["target_frames"]))
    print("   图像尺寸: {}".format(config["image_size"]))
    print("   基础通道: {}".format(config["base_channels"]))

    # 随机测试
    print()
    print("-" * 50)
    test_random_input(model, config)

    print()
    print("=" * 50)
    print("测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    test_model()