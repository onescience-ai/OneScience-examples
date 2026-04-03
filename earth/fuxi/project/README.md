# FuXi: A Cascade Machine Learning Forecasting System

## Overview

This is an implementation of the FuXi weather forecasting model based on the paper:
"FuXi: a cascade machine learning forecasting system for 15-day global weather forecast"

## Model Architecture

The FuXi model consists of three main components:

1. **Cube Embedding**: 3D patch embedding that reduces spatial-temporal dimensions
2. **U-Transformer**: U-shaped architecture with Swin Transformer V2 blocks
3. **FC Layer**: Fully connected layer for final prediction

## Project Structure

```
fuxi/
├── config.py          # Configuration parameters
├── model.py           # FuXi model definition
├── train.py           # Training script
├── eval.py            # Evaluation script
└── README.md          # This file
```

## Installation

```bash
pip install torch numpy
```

## Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
python eval.py
```

## Configuration

Key parameters in `config.py`:

- `img_size`: Input image size (T, Lat, Lon) = (2, 721, 1440)
- `patch_size`: Patch size for embedding = (2, 4, 4)
- `in_chans`: Number of input channels = 70
- `embed_dim`: Embedding dimension = 1536
- `depth`: Number of transformer layers = 48
- `num_heads`: Number of attention heads = 8

## Components Used

This implementation uses the following components from `onescience.modules`:

- `FuxiEmbedding`: 3D patch embedding
- `FuxiTransformer`: U-shaped transformer backbone
- `FuxiFC`: Fully connected output layer

## References

Chen, L., Zhong, X., Zhang, F., Cheng, Y., Xu, Y., Qi, Y., & Li, H. (2023). 
FuXi: a cascade machine learning forecasting system for 15-day global weather forecast. 
npj Climate and Atmospheric Science, 6(1), 1-12.
