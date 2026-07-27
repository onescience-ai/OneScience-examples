import sys
sys.path.insert(0, '/root/private_data/pzy/lilith')

from inference.simple_forecaster import SimpleForecaster
import json
from datetime import datetime

# 加载模型
print("🔮 加载模型...")
forecaster = SimpleForecaster(
    checkpoint_path="checkpoints/lilith_best.pt",
    device="auto",
    enable_live_data=True
)

# 生成预测
print(f"📍 预测位置: 40.7128, -74.006 (纽约)")
print(f"📅 预测天数: 90 天")
print("⏳ 生成预测中...")

response = forecaster.forecast(
    latitude=40.7128,
    longitude=-74.006,
    forecast_days=90
)

# 保存结果
output_file = f"forecast_nyc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w") as f:
    json.dump(response, f, indent=2, default=str)

print(f"\n✅ 预测完成！")
print(f"📁 结果已保存到: {output_file}")

# 显示前5天的预测
if isinstance(response, dict) and "forecasts" in response:
    print(f"\n📊 前5天预测预览:")
    print("-" * 60)
    for i, day in enumerate(response["forecasts"][:5]):
        print(f"  {day.get('date', 'N/A')}: "
              f"最高 {day.get('temperature_max', 'N/A')}°C, "
              f"最低 {day.get('temperature_min', 'N/A')}°C, "
              f"降水 {day.get('precipitation', 'N/A')}mm")
