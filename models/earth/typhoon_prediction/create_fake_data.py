import os
import numpy as np
import xarray as xr

# 虚拟数据集存放路径（固定这个路径，后面config.py直接填写）
fake_data_path = "/root/private_data/LGS/fake_typhoon_dataset"
os.makedirs(fake_data_path, exist_ok=True)

# 模拟参数，和模型配置匹配：224x224分辨率
sample_num = 200  # 生成200条虚拟样本，足够测试流程
H, W = 224, 224

for idx in range(sample_num):
    # 模拟气压场数据 pressure
    pressure = np.random.normal(loc=101300, scale=500, size=(H, W)).astype(np.float32)
    ds = xr.Dataset(
        {
            "pressure": xr.DataArray(pressure, dims=["lat", "lon"])
        }
    )
    # 保存nc文件，气象数据集标准格式
    ds.to_netcdf(os.path.join(fake_data_path, f"sample_{idx:03d}.nc"))

print(f"✅虚拟数据集生成完成！路径：{fake_data_path}")