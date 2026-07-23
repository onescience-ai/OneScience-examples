---
license: apache-2.0
language:
- es
- en
metrics:
- mse
pipeline_tag: image-to-image
tags:
- climate
- transformers
---


# Europe Reanalysis Super Resolution

The aim of the project is to create a Machine learning (ML) model that can generate high-resolution regional reanalysis data (similar to the one produced by CERRA) by downscaling global reanalysis data from ERA5.


This will be accomplished by using state-of-the-art Deep Learning (DL) techniques like U-Net, conditional GAN, and diffusion models (among others). Additionally, an ingestion module will be implemented to assess the possible benefit of using CERRA pseudo-observations as extra predictors. Once the model is designed and trained, a detailed validation framework takes the place.


It combines classical deterministic error metrics with in-depth validations, including time series, maps, spatio-temporal correlations, and computer vision metrics, disaggregated by months, seasons, and geographical regions, to evaluate the effectiveness of the model in reducing errors and representing physical processes. This level of granularity allows for a more comprehensive and accurate assessment, which is critical for ensuring that the model is effective in practice.


Moreover, tools for interpretability of DL models can be used to understand the inner workings and decision-making processes of these complex structures by analyzing the activations of different neurons and the importance of different features in the input data.

This work is funded by [Code for Earth 2023](https://codeforearth.ecmwf.int/) initiative.
