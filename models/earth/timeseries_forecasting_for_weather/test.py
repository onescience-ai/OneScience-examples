#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import tensorflow as tf

# 1. 读取数据
csv_path = "jena_climate_2009_2016.csv"
df = pd.read_csv(csv_path)
print("成功读取数据")


# In[15]:


# 2. 选择特征（与文档中的表格对应：+ 表示选中，包括 p, T, VPmax, VPdef, sh, rho, wv）
selected_features = [
    "p (mbar)",
    "T (degC)",
    "VPmax (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "rho (g/m**3)",
    "wv (m/s)",
]
features = df[selected_features]
features.index = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
print("成功读取特征")


# In[ ]:


# 3. 数据划分 (71.5%训练集)
features = features.iloc[:5000] # 只取前面5000份数据集进行测试模型
split_fraction = 0.715
train_split = int(len(features) * split_fraction)


# In[17]:


# 5. 构造时间序列数据集
# 参数配置：
# past = 720 (过去 720 个 10分钟点 = 120小时)
# step = 6 (采样率：每 6 个点取 1 个，即每小时 1 个点)
# delay = 72 (预测未来第 72 个点 = 12小时后)
past = 720
step = 6
delay = 72
start = past + delay
end = train_split


# In[18]:


# 训练集
x_train = features.iloc[:train_split].values
y_train = features.iloc[start:train_split]["T (degC)"].values

train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=past // step,
    sampling_rate=step,
    batch_size=256,
)

# 验证集
x_end = len(features) - delay - 1
x_val = features.iloc[train_split:].values
y_val = features.iloc[train_split + start :]["T (degC)"].values

val_dataset = tf.keras.utils.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=past // step,
    sampling_rate=step,
    batch_size=256,
)


# In[ ]:


# 6. 构建 LSTM 模型
model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(
            32, input_shape=(train_dataset.element_spec[0].shape[1], train_dataset.element_spec[0].shape[2])
        ),
        tf.keras.layers.Dense(1),
    ]
)

# 7. 编译与训练
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mae",
    metrics=[
        "mse",  # Mean Squared Error (均方误差)
        tf.keras.metrics.RootMeanSquaredError(name="rmse")  # Root Mean Squared Error (均方根误差)
    ]
)


# In[ ]:


print("开始训练")
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
)

val_results = model.evaluate(val_dataset, verbose=0)
print(f"验证集 MAE (平均绝对误差): {val_results[0]:.4f}")
print(f"验证集 MSE (均方误差): {val_results[1]:.4f}")
print(f"验证集 RMSE (均方根误差): {val_results[2]:.4f}")

