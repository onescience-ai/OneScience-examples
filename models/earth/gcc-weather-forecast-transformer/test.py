from huggingface_hub import hf_hub_download
import torch
import pickle
import sys

# 1. Download Model Artifacts
repo_id = "assix-research/gcc-weather-forecast-transformer"
model_path = hf_hub_download(repo_id=repo_id, filename="weather_transformer.pt")
scaler_path = hf_hub_download(repo_id=repo_id, filename="feature_scaler.pkl")
code_path = hf_hub_download(repo_id=repo_id, filename="model.py")

# 2. Load Architecture Dynamically
import importlib.util
spec = importlib.util.spec_from_file_location("weather_model", code_path)
weather_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weather_model)

# 3. Initialize & Load Weights
model = weather_model.WeatherTransformer(input_dim=4, seq_len=72)
model.load_state_dict(torch.load(model_path))
model.eval()

print("✅ Model loaded successfully!")
print(model)

# ============ Random Data Test Prediction with Metrics ============
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Generate synthetic weather data with a "future" target
def generate_random_weather_data_with_target(seq_len=72):
    """
    Generate synthetic weather data with a known next-hour target.
    Returns: (historical_data, target)
        historical_data: 72 hours of data (72, 4)
        target: next hour's temperature (scalar)
    """
    # Generate 73 hours so we have a known next hour
    hours = np.arange(seq_len + 1)
    daily_cycle = np.sin(hours / 24 * 2 * np.pi) * 8
    temp = 30 + daily_cycle + np.random.randn(seq_len + 1) * 2
    humidity = np.clip(50 - daily_cycle * 1.5 + np.random.randn(seq_len + 1) * 5, 10, 90)
    pressure = 1010 + np.random.randn(seq_len + 1) * 3
    wind = np.abs(np.random.randn(seq_len + 1) * 8 + 10)
    
    full_data = np.column_stack([temp, humidity, pressure, wind])
    
    # First 72 hours = input
    historical = full_data[:seq_len]
    # Last hour = target (what we want to predict)
    target = full_data[seq_len]
    
    return historical, target

# ============ Run Multiple Predictions for Metrics ============
print("\n" + "="*60)
print("🌤️ Weather Prediction with Performance Metrics")
print("="*60)

# Settings
n_tests = 50  # Number of test runs for metrics
predictions = []
targets = []
predictions_normalized = []
targets_normalized = []

print(f"\n📊 Running {n_tests} test predictions...")

for i in range(n_tests):
    # Generate data with known target
    random_data, target = generate_random_weather_data_with_target(72)
    
    # Normalize input
    scaled_data = scaler.transform(random_data)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        pred_norm = model(input_tensor).item()
    
    # Inverse transform: only temperature
    pred_array = np.zeros((1, 4))
    pred_array[0, 0] = pred_norm
    pred_array[0, 1:] = 0
    pred_physical = scaler.inverse_transform(pred_array)[0][0]
    
    # Target is already in physical units
    target_physical = target[0]  # Temperature only
    
    # Store results
    predictions.append(pred_physical)
    targets.append(target_physical)
    predictions_normalized.append(pred_norm)
    targets_normalized.append(scaler.transform(target.reshape(1, -1))[0][0])

# Convert to numpy arrays
predictions = np.array(predictions)
targets = np.array(targets)

# ============ Calculate Metrics ============
print("\n" + "="*60)
print("📊 Performance Metrics (Temperature Prediction)")
print("="*60)

# 1. MAE - Mean Absolute Error
mae = mean_absolute_error(targets, predictions)
print(f"\n📏 MAE (Mean Absolute Error): {mae:.4f} °C")
print(f"   → Average prediction error: ±{mae:.2f}°C")

# 2. RMSE - Root Mean Square Error
rmse = np.sqrt(mean_squared_error(targets, predictions))
print(f"\n📐 RMSE (Root Mean Square Error): {rmse:.4f} °C")
print(f"   → Penalizes larger errors more heavily")

# 3. Bias - Systematic error
bias = np.mean(predictions - targets)
print(f"\n⚖️  Bias: {bias:+.4f} °C")
print(f"   → {'Overestimates' if bias > 0 else 'Underestimates'} temperatures by {abs(bias):.2f}°C on average")

# 4. R² Score - Coefficient of determination
r2 = r2_score(targets, predictions)
print(f"\n📈 R² Score: {r2:.4f}")
print(f"   → {r2*100:.1f}% of temperature variance explained by the model")

# 5. Correlation
correlation = np.corrcoef(targets, predictions)[0, 1]
print(f"\n🔗 Correlation: {correlation:.4f}")
print(f"   → {'Strong' if correlation > 0.8 else 'Moderate' if correlation > 0.5 else 'Weak'} positive correlation")

# 6. Max Absolute Error
max_abs_error = np.max(np.abs(predictions - targets))
print(f"\n⚠️  Max Absolute Error: {max_abs_error:.4f} °C")
print(f"   → Worst single prediction error")

# 7. Percentage of predictions within 2°C
within_2deg = np.mean(np.abs(predictions - targets) <= 2.0) * 100
print(f"\n🎯 Predictions within ±2°C: {within_2deg:.1f}%")
within_1deg = np.mean(np.abs(predictions - targets) <= 1.0) * 100
print(f"   Predictions within ±1°C: {within_1deg:.1f}%")

# 8. Standard Deviation of errors
std_error = np.std(predictions - targets)
print(f"\n📊 Error Std Deviation: {std_error:.4f} °C")

# ============ Summary Table ============
print("\n" + "="*60)
print("📋 Metrics Summary")
print("="*60)
print(f"{'Metric':<25} {'Value':>15} {'Interpretation':>20}")
print("-"*60)
print(f"{'MAE':<25} {mae:>15.4f}°C {'Avg error magnitude':>20}")
print(f"{'RMSE':<25} {rmse:>15.4f}°C {'Penalizes large errors':>20}")
print(f"{'Bias':<25} {bias:>+15.4f}°C {'Systematic offset':>20}")
print(f"{'R²':<25} {r2:>15.4f} {'Variance explained':>20}")
print(f"{'Correlation':<25} {correlation:>15.4f} {'Trend alignment':>20}")
print(f"{'Max Abs Error':<25} {max_abs_error:>15.4f}°C {'Worst-case error':>20}")
print(f"{'Std of Errors':<25} {std_error:>15.4f}°C {'Error spread':>20}")
print(f"{'Within ±2°C':<25} {within_2deg:>14.1f}% {'Accuracy threshold':>20}")
print("="*60)

# ============ Show Sample Predictions ============
print("\n" + "="*60)
print("🔍 Sample Predictions (First 10 test cases)")
print("="*60)
print(f"{'#':<5} {'Actual (°C)':>12} {'Predicted (°C)':>14} {'Error (°C)':>12}")
print("-"*60)
for i in range(min(10, n_tests)):
    error = predictions[i] - targets[i]
    print(f"{i+1:<5} {targets[i]:>12.2f} {predictions[i]:>14.2f} {error:>+12.2f}")
print("="*60)

print("\n✅ Test complete!")

# Explanation of metrics
print("\n" + "="*60)
print("📖 Metric Interpretation Guide")
print("="*60)
print("""
🔹 MAE (Mean Absolute Error): Average absolute difference between predictions and actual values.
   Lower is better. | MAE < 1°C is generally considered excellent.

🔹 RMSE (Root Mean Square Error): Similar to MAE but penalizes large errors more.
   Lower is better. RMSE ≥ MAE (due to squaring).

🔹 Bias: Systematic overestimation (+) or underestimation (-).
   Closer to 0 is better. |Bias| < 0.5°C indicates unbiased predictions.

🔹 R² (R-squared): Proportion of variance in the target explained by the model.
   Ranges from -∞ to 1. Closer to 1 is better. R² > 0.9 is excellent.

🔹 Correlation: Linear relationship strength between predictions and actuals.
   Ranges from -1 to 1. Closer to 1 is better.

🔹 Max Absolute Error: Largest single prediction error.
   Useful for detecting outliers.

🔹 Within ±2°C: Percentage of predictions within 2°C of the actual value.
   Higher is better (target > 90% for practical applications).
""")
print("="*60)
