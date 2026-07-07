---
title: GCC Weather Forecast Transformer
emoji: 🌦️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: true
license: mit
tags:
  - time-series
  - weather
  - transformer
  - pytorch
  - climate
---

# 🌦️ GCC Weather Forecast Transformer

A specialized Transformer-based model designed for short-horizon meteorological forecasting in the Gulf Cooperation Council (GCC) region. Trained on NVIDIA DGX Spark infrastructure with Grace Blackwell GPU, this model leverages high-resolution historical data to predict immediate future weather conditions for **Dubai** and **Riyadh**.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/assix-research/gcc-weather-forecast-transformer)

## 🎮 Live Demo

👉 **Try the model here:** [GCC Weather Forecast Transformer Space](https://huggingface.co/spaces/assix-research/gcc-weather-forecast-transformer)

* **Real-time Inference**: Generates forecasts using live data fetched on-demand.
* **Supported Cities**:
    * **Dubai**, United Arab Emirates 🇦🇪
    * **Riyadh**, Saudi Arabia 🇸🇦

---

## ⚡ Technical Summary

| Attribute | Specification |
| :--- | :--- |
| **Architecture** | Transformer Encoder (Sequence-to-Vector) |
| **Input Context** | 72 Hours (Sliding Window) |
| **Forecast Horizon** | 1 Hour Ahead |
| **Variables** | Temperature, Humidity, Pressure (MSL), Wind Speed |
| **Training Hardware** | NVIDIA DGX Spark (Grace Blackwell / A100 optimized) |
| **Data Source** | Open-Meteo Historical Archive |

---

## 🚀 Quick Start (Python)

You can download and use this model directly in your Python environment.

```python
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
```


---

## 🧠 Model Architecture

The core is a compact Transformer Encoder optimized for structured time-series data. Unlike LSTM or RNN baselines, the Transformer mechanism allows the model to attend to the entire 72-hour history simultaneously, capturing long-range dependencies (e.g., diurnal cycles and pressure fronts).
Hyperparameters

    Embedding Dimension (dmodel​): 256

    Attention Heads: 8 (Multi-Head Self-Attention)

    Layers: 4 Encoder Blocks

    Feedforward Network: 512 dimensions

    Dropout: 0.1

    Positional Encoding: Sinusoidal injection to preserve temporal order.

Inputs & Outputs

    Input X: Tensor of shape (Batch, 72, 4).

        Features: temperature_2m, relative_humidity_2m, pressure_msl, wind_speed_10m.

    Output y: Tensor of shape (Batch, 4).

        Predicted values for the very next hour.


---

## 🛠️ Data Pipeline

The system operates on a rigorous pipeline ensuring data consistency between training and inference.

    Ingestion:

        Fetches hourly historical data from the Open-Meteo API.

        Locations: Dubai (25.2°N, 55.2°E) and Riyadh (24.7°N, 46.6°E).

    Preprocessing:

        Time Alignment: All timestamps standardized to UTC.

        Normalization: Scikit-Learn StandardScaler (Z-score normalization) fit on the training corpus.

    Inference:

        The model accepts normalized tensors.

        Outputs are inverse-transformed back to physical units (°C, %, hPa, km/h) for display.


---


## 👨‍💻 Attribution & Credits

* **Developed by:** Assix Research (2026)
* **Compute:** Trained on NVIDIA DGX Spark infrastructure.
* **Data Provider:** [Open-Meteo](https://open-meteo.com/)