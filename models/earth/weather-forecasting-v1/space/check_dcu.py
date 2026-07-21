#!/usr/bin/env python
import subprocess
import re

def get_dcu_info():
    try:
        # 执行 hy-smi 命令
        result = subprocess.run(['hy-smi'], capture_output=True, text=True)
        output = result.stdout
        
        # 提取关键数据（简单粗暴）
        print("="*50)
        print("DCU 状态检查器 (基于 hy-smi)")
        print("="*50)
        
        # 找温度
        temp_match = re.search(r'(\d+\.\d+)°C', output)
        if temp_match:
            print(f"🌡️  当前温度: {temp_match.group(1)}°C")
        
        # 找功耗
        power_match = re.search(r'(\d+\.\d+)W', output)
        if power_match:
            print(f"⚡ 当前功耗: {power_match.group(1)}W")
        
        # 找显存使用率
        vram_match = re.search(r'(\d+)%\s+(\d+\.\d+)%\s+(\w+)', output)
        if vram_match:
            print(f"💾 显存使用率: {vram_match.group(1)}%")
            print(f"🧮 计算核心使用率: {vram_match.group(2)}%")
            print(f"📊 运行模式: {vram_match.group(3)}")
        
        # 检查是否有错误
        if "Error" in output or "Fail" in output:
            print("⚠️  检测到异常，请检查 DCU 驱动状态")
        else:
            print("✅ DCU 工作正常，驱动无报错")
        
        print("="*50)
        
    except FileNotFoundError:
        print("❌ 未找到 hy-smi 命令，请确认 DCU 驱动已安装")
    except Exception as e:
        print(f"❌ 执行出错: {e}")

if __name__ == "__main__":
    get_dcu_info()
