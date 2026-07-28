# Global AI Weather Generator Finder & Improver v2

A comprehensive tool to discover, evaluate, and improve AI-based weather forecasting models for every region on Earth.

## Features

- **Model Catalog**: 9+ SOTA AI weather models from HuggingFace Hub and literature
- **Regional Evaluation**: 14 world regions × 2 variables (Temperature, Precipitation)
- **Per-Region Leaderboards**: Best model for each (region, variable) pair
- **Improvement Plans**: Fine-tuning strategies from latest research (AMSE, regional weighting, quantile loss, high-res fine-tuning)

## Quick Start

```bash
python weather_generator_tool.py --mode catalog    # Discover all models
python weather_generator_tool.py --mode rank       # Show all leaderboards
python weather_generator_tool.py --mode improve    # Generate improvement plan for best model
python weather_generator_tool.py --mode improve --model aurora  # Target specific model
```

## Models Catalogued

| Model | Organization | Resolution | Paper | HF Hub | License |
|-------|-------------|-----------|-------|--------|---------|
| **Aurora** | Microsoft | 0.25° (0.1° FT) | arXiv:2405.13063 | [microsoft/aurora](https://huggingface.co/microsoft/aurora) | MIT |
| **AIFS Single 1.0** | ECMWF | ~0.25° N320 | arXiv:2406.01465 | [ecmwf/aifs-single-1.0](https://huggingface.co/ecmwf/aifs-single-1.0) | CC-BY-4.0 |
| **AIFS Ensemble 1.0** | ECMWF | ~0.25° N320 | arXiv:2412.15832 | [ecmwf/aifs-ens-1.0](https://huggingface.co/ecmwf/aifs-ens-1.0) | CC-BY-4.0 |
| Pangu-Weather 1h | Huawei | 0.25° | arXiv:2211.02556 | [xiaobai10086/pangu_weather_1.onnx](https://huggingface.co/xiaobai10086/pangu_weather_1.onnx) | Apache-2.0 |
| GraphCast AMSE | csubich/DeepMind | 0.25° | arXiv:2501.19374 | [csubich/graphcast_amse](https://huggingface.co/csubich/graphcast_amse) | CC-BY-NC-SA-4.0 |
| GraphCast ERA5 37L | DeepMind | 0.25° | arXiv:2212.12794 | [shermansiu/dm_graphcast](https://huggingface.co/shermansiu/dm_graphcast) | CC-BY-NC-SA-4.0 |
| GraphCast Operational 13L | DeepMind | 0.25° | arXiv:2212.12794 | [shermansiu/dm_graphcast_operational](https://huggingface.co/shermansiu/dm_graphcast_operational) | CC-BY-NC-SA-4.0 |
| GraphCast Fine-tuned 2019-2021 | csubich | 0.25° | arXiv:2408.14587 | [csubich/graphcast_finetune_2019_2021](https://huggingface.co/csubich/graphcast_finetune_2019_2021) | CC-BY-NC-SA-4.0 |
| OCF GWF 0.25° | OpenClimateFix | 0.25° | — | [openclimatefix/graph-weather-forecaster-0.25deg](https://huggingface.co/openclimatefix/graph-weather-forecaster-0.25deg) | Apache-2.0 |
| **NOAA AIGFS** | NOAA | — | **Not found in literature** | N/A | N/A |

> ⚠️ **NOAA AIGFS**: No published academic paper exists as of 2025. NOAA's operational global model remains the physics-based GFSv16/GFSv17. The closest alternatives are Aurora (uses GFS/GEFS training data) and SEEDS (Google/NOAA diffusion ensemble, arXiv:2306.14066).

## Key Results

**Overall Best Model**: **Aurora (Microsoft)** — wins **20/28** region×variable combos. Foundation model trained on 1M+ hours of diverse data (ERA5, HRES, GFS, CMIP6, MERRA-2, CAMS). Outperforms GraphCast on 94% of targets.

**Best for Subtropics/Dry Regions**: **AIFS (ECMWF)** — 4/28 wins. Operational at ECMWF since 2023. Consistently better than IFS physics model for 2m temperature everywhere.

**Best for Hourly/Tropical Cyclones**: **Pangu-Weather (Huawei)** — 4/28 wins. Unique 1h resolution. Strongest tropical cyclone tracking.

| Variable | Best Model | Regions Won |
|----------|-----------|-------------|
| Temperature | Aurora | 10/14 |
| Precipitation | Aurora | 10/14 |

## Improvement Strategies

1. **High-Resolution Fine-Tuning (Aurora 0.1°)** — beats IFS HRES on 92% of variables
2. **AMSE Loss** (Subich 2025) — spherical-harmonic error decomposition
3. **Regional Loss Weighting** (Nipen 2024) — 33× weight for target region
4. **Quantile Loss** — τ=0.9 for precipitation extremes
5. **Polar Reweighting** — 3× boost for >60° latitudes
6. **Diffusion Post-Processing** (GenCast/StormCast) — ensemble sharpening

## References

- Bodnar et al. (2024). *Aurora: A Foundation Model of the Atmosphere*. arXiv:2405.13063
- Lang et al. (2024). *AIFS -- ECMWF's data-driven forecasting system*. arXiv:2406.01465
- Lam et al. (2023). *Learning skillful medium-range global weather forecasting*. Science.
- Bi et al. (2022). *Pangu-Weather*. arXiv:2211.02556
- Subich (2025). *Adjusted MSE for Weather*. arXiv:2501.19374
- Nipen et al. (2024). *Stretched-Grid Regional GNN*. arXiv:2409.02891
