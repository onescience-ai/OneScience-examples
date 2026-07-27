import pickle
import torch
import numpy as np
import xarray as xr
import pandas as pd
import os
import shutil
from datetime import datetime
from tqdm import tqdm

# ============================================================================
# 工具函数
# ============================================================================

def load_info():
    """加载info.pkl文件"""
    if os.path.exists('./info.pkl'):
        with open('./info.pkl', 'rb') as f:
            info = pickle.load(f)
        print("✓ info.pkl 加载成功")
        return info
    else:
        print("✗ info.pkl 不存在")
        return None

def inspect_info(info):
    """检查info.pkl的内容"""
    print("\n" + "="*60)
    print("info.pkl 内容")
    print("="*60)
    
    if info is None:
        return None, None
    
    print(f"类型: {type(info)}")
    if isinstance(info, dict):
        print(f"键: {list(info.keys())}")
        
        # 提取变量信息
        variables_info = None
        if 'variables' in info:
            variables_info = info['variables']
            if isinstance(variables_info, dict):
                var_names = list(variables_info.keys())
                print(f"\n变量数量: {len(var_names)}")
                print(f"前10个变量: {var_names[:10]}")
                
                # 检查是否有dim信息
                if 'dims' in variables_info:
                    print(f"维度信息: {variables_info['dims']}")
                
                return var_names, variables_info
            elif isinstance(variables_info, list):
                print(f"\n变量列表 (前10个): {variables_info[:10]}")
                return variables_info, None
        
        # 检查其他可能的键
        if 'input_variables' in info:
            print(f"输入变量: {len(info['input_variables'])}个")
        if 'output_variables' in info:
            print(f"输出变量: {len(info['output_variables'])}个")
        if 'input_shape' in info:
            print(f"输入形状: {info['input_shape']}")
        if 'output_shape' in info:
            print(f"输出形状: {info['output_shape']}")
    
    return None, None

def load_model(device="cuda:0"):
    """加载模型"""
    model_path = "./gamfs_3h_traced.pt"
    
    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        return None
    
    print(f"\n加载模型: {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        if device != "cpu":
            model = model.to(device)
        print("✓ 模型加载成功")
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

def test_model_inputs(model, device="cuda:0"):
    """测试不同输入尺寸，找到正确的配置"""
    print("\n" + "="*60)
    print("测试模型输入尺寸")
    print("="*60)
    
    # 从info.pkl获取变量数量
    info = load_info()
    var_names, _ = inspect_info(info)
    
    if var_names:
        n_vars = len(var_names)
        print(f"\n从info.pkl获取变量数量: {n_vars}")
    else:
        # 默认值
        n_vars = 54
        print(f"\n使用默认变量数量: {n_vars}")
        var_names = [f'var_{i:02d}' for i in range(n_vars)]
    
    # 测试不同的空间尺寸
    test_sizes = [
        (256, 256),
        (288, 288),
        (320, 320),
        (360, 360),
        (384, 384),
        (448, 448),
        (480, 480),
        (512, 512),
        (576, 576),
        (640, 640),
        (720, 720),
    ]
    
    print(f"\n测试变量数: {n_vars}")
    print("测试空间尺寸...")
    
    for h, w in test_sizes:
        try:
            print(f"\n  测试: ({n_vars}, {h}, {w})")
            
            # 创建测试输入
            test_input = torch.randn(1, n_vars, h, w).to(device).float()
            time_tag = torch.tensor([0.5, 0.5, 0.5]).to(device).unsqueeze(0)
            
            with torch.no_grad():
                output = model(test_input, time_tag)
            
            print(f"  ✓ 成功! 输出形状: {output.shape}")
            print(f"  ✓ 推荐配置: 变量数={n_vars}, 高度={h}, 宽度={w}")
            print(f"  输出形状: {output.shape}")
            
            # 检查输出变量数
            if len(output.shape) >= 2:
                out_vars = output.shape[1] if len(output.shape) > 1 else 1
                print(f"  输出变量数: {out_vars}")
            
            return True, n_vars, h, w, var_names, output.shape
            
        except Exception as e:
            error_msg = str(e)
            if 'H: 0' in error_msg or 'W: 0' in error_msg:
                print(f"  ✗ 维度错误: {error_msg[:80]}...")
            elif 'channels' in error_msg:
                print(f"  ✗ 通道错误: {error_msg[:80]}...")
            else:
                print(f"  ✗ 失败: {error_msg[:80]}...")
            continue
    
    print("\n✗ 所有尺寸测试都失败了")
    return False, None, None, None, None, None

def create_virtual_data(n_vars, h, w, var_names):
    """创建虚拟数据"""
    print("\n" + "="*60)
    print("创建虚拟数据")
    print("="*60)
    
    # 创建数据
    data = np.random.randn(n_vars, h, w).astype(np.float32) * 0.1
    
    # 创建经纬度
    lon = np.linspace(-180, 180, w, dtype=np.float32)
    lat = np.linspace(-90, 90, h, dtype=np.float32)
    
    # 确保变量名数量匹配
    if len(var_names) < n_vars:
        var_names = var_names + [f'var_{i:02d}' for i in range(len(var_names), n_vars)]
    elif len(var_names) > n_vars:
        var_names = var_names[:n_vars]
    
    # 转换为tensor
    data_tensor = torch.from_numpy(data).to(torch.float16).unsqueeze(0).float()
    
    print(f"数据形状: {data_tensor.shape}")
    print(f"变量数量: {len(var_names)}")
    print(f"空间维度: {h} x {w}")
    
    return data_tensor, var_names, lat, lon

def save_output(output, var_names, time, lat, lon, save_path):
    """保存输出为NetCDF文件"""
    print(f"  输出形状: {output.shape}")
    print(f"  lat形状: {lat.shape}")
    print(f"  lon形状: {lon.shape}")
    
    # 获取输出的空间维度
    if output.ndim == 3:
        # 输出是 (vars, h, w)
        out_vars, out_h, out_w = output.shape
    elif output.ndim == 4:
        # 输出是 (batch, vars, h, w)
        out_vars, out_h, out_w = output.shape[1], output.shape[2], output.shape[3]
    else:
        raise ValueError(f"不支持的输出维度: {output.ndim}")
    
    print(f"  输出变量数: {out_vars}, 高度: {out_h}, 宽度: {out_w}")
    
    # 如果经纬度维度不匹配，重新创建
    if len(lat) != out_h:
        print(f"  重新创建纬度: {out_h}")
        lat = np.linspace(-90, 90, out_h, dtype=np.float32)
    if len(lon) != out_w:
        print(f"  重新创建经度: {out_w}")
        lon = np.linspace(-180, 180, out_w, dtype=np.float32)
    
    # 确保变量名数量匹配
    if len(var_names) < out_vars:
        var_names = var_names + [f'var_{i:02d}' for i in range(len(var_names), out_vars)]
    elif len(var_names) > out_vars:
        var_names = var_names[:out_vars]
    
    # 提取数据
    if output.ndim == 4:
        data_to_save = output[0]  # 去掉batch维度
    else:
        data_to_save = output
    
    # 创建数据变量
    data_vars = {}
    for i, var_name in enumerate(var_names):
        data_vars[var_name] = xr.DataArray(
            data_to_save[i, :, :],
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon},
            attrs={'units': '1', 'long_name': var_name}
        )
    
    # 创建数据集
    ds = xr.Dataset(
        data_vars,
        attrs={
            'title': 'AI-GAMFS Virtual Test',
            'institution': 'Test',
            'source': 'Virtual Data',
            'creation_time': str(time),
            'description': 'Generated for testing purposes'
        }
    )
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ds.to_netcdf(save_path)
    print(f"  ✓ 已保存: {os.path.basename(save_path)}")
    return ds

def rolling_predict_simple(init_field, target_step, model, device, start_time):
    """简化版滚动预测，只使用3小时模型"""
    output = init_field
    current_time = start_time
    
    for step in range(target_step):
        # 创建时间特征
        time_tag = torch.tensor([
            current_time.hour / 24,
            current_time.weekday() / 7,
            current_time.timetuple().tm_yday / 365.25
        ]).to(device).unsqueeze(0)
        
        # 预测
        output = model(output.to(device).float(), time_tag)
        current_time = current_time + pd.Timedelta(hours=3)
    
    return output.detach().cpu().numpy().squeeze()

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("="*60)
    print("AI-GAMFS 完整测试程序")
    print("="*60)
    
    # 1. 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 2. 加载模型
    model = load_model(device)
    if model is None:
        print("模型加载失败，退出")
        return
    
    # 3. 测试输入尺寸
    success, n_vars, h, w, var_names, output_shape = test_model_inputs(model, device)
    if not success:
        print("\n无法找到正确的输入配置，退出")
        return
    
    # 4. 从测试结果获取输出维度
    if output_shape is not None:
        if len(output_shape) == 4:
            out_vars, out_h, out_w = output_shape[1], output_shape[2], output_shape[3]
        elif len(output_shape) == 3:
            out_vars, out_h, out_w = output_shape[0], output_shape[1], output_shape[2]
        else:
            out_h, out_w = h, w
    else:
        out_h, out_w = h, w
    
    print(f"\n输出维度: 变量={out_vars}, 高度={out_h}, 宽度={out_w}")
    
    # 5. 创建虚拟数据（使用输入维度）
    init_field, var_names, lat, lon = create_virtual_data(n_vars, h, w, var_names)
    
    # 6. 设置时间
    start_time = pd.Timestamp("2024-07-26 00:00:00")
    init_time = start_time
    
    print(f"\n开始时间: {init_time}")
    
    # 7. 创建输出文件夹
    time_key = start_time.strftime("%Y%m%d_%H%M")
    output_folder = f"./inference/{time_key}/"
    os.makedirs(output_folder, exist_ok=True)
    
    # 8. 执行预测
    num_steps = 5
    print(f"\n开始预测 (共{num_steps}步)")
    print("="*60)
    
    for i in tqdm(range(num_steps), desc="预测步数"):
        target_step = i + 1
        
        try:
            print(f"\n步数 {target_step}:")
            
            # 预测
            output = rolling_predict_simple(
                init_field.clone(),
                target_step,
                model,
                device,
                start_time
            )
            
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
            
            # 保存
            save_time = start_time + pd.Timedelta(hours=target_step * 3)
            save_path = f"{output_folder}/AI_GAMFS.{start_time.strftime('%Y%m%d_%H%M')}+{save_time.strftime('%Y%m%d_%H%M')}.V01.nc"
            
            # 注意：这里使用输出维度创建经纬度
            out_lat = np.linspace(-90, 90, output.shape[1], dtype=np.float32)
            out_lon = np.linspace(-180, 180, output.shape[2], dtype=np.float32)
            
            save_output(output, var_names, save_time, out_lat, out_lon, save_path)
            
        except Exception as e:
            print(f"  步数 {target_step} 失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 8. 清理
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(f"所有处理完成！")
    print(f"输出目录: {output_folder}")
    print("="*60)

if __name__ == "__main__":
    main()