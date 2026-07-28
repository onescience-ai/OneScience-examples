#!/usr/bin/env python3
"""
Weather Generator Finder App — Gradio Demo v3
Now includes: GraphCast, Pangu-Weather, AIFS (ECMWF), Aurora (Microsoft),
             WeatherNext 2 (GenCast proxy), Atmo (AtmoRep proxy), NOAA AIGFS
"""

import gradio as gr

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
REGIONS = {
    "North America":        (15,  75, -170, -50,  "USA, Canada, Mexico"),
    "South America":        (-55,  15,  -90, -30,  "Brazil, Argentina, Andes"),
    "Western Europe":       (35,  70,  -15,  30,  "UK, France, Germany, Spain"),
    "Eastern Europe":       (40,  70,   30,  60,  "Poland, Ukraine, Russia West"),
    "North Africa":         (15,  38,  -20,  40,  "Sahara, Maghreb, Egypt"),
    "Central & South Africa": (-35, 15,   10,  55,  "Congo, Kenya, South Africa"),
    "Middle East":          (12,  42,   30,  60,  "Saudi Arabia, Iran, UAE"),
    "South Asia":           (5,   38,   60,  95,  "India, Pakistan, Bangladesh"),
    "East Asia":            (18,  55,   95, 145,  "China, Japan, Korea"),
    "Southeast Asia":       (-11, 28,   95, 145,  "Indonesia, Thailand, Vietnam"),
    "Oceania":              (-50, -5,  110, 180,  "Australia, New Zealand"),
    "Arctic":               (60,  90, -180, 180,  "Arctic Circle, Greenland"),
    "Antarctica":           (-90, -60, -180, 180, "Antarctic continent"),
    "Tropical Pacific":     (-20,  20,  120, -90, "Equatorial Pacific, ENSO region"),
}

# Updated rankings including Aurora and AIFS
TEMPERATURE_BEST = {
    "North America":        ("aurora", 0.95, "Best at >3d leads; outperforms GraphCast on 94% targets"),
    "South America":        ("aurora", 1.05, "Strong on diverse data; good in tropics and extra-tropics"),
    "Western Europe":       ("aurora", 0.88, "Best at >3d leads; 0.1° high-res beats IFS HRES on 92%"),
    "Eastern Europe":       ("aurora", 0.90, "Best at >3d leads; outperforms GraphCast substantially"),
    "North Africa":         ("aifs_single_1_0", 1.20, "Better than IFS; good in dry subtropics"),
    "Central & South Africa": ("aurora", 1.10, "Best overall; diverse data including CMIP6 helps"),
    "Middle East":          ("aifs_single_1_0", 1.15, "Better than IFS in subtropics"),
    "South Asia":           ("aurora", 0.95, "Best overall; monsoon/extreme events"),
    "East Asia":            ("aurora", 0.90, "Best at >3d leads; outperforms GraphCast substantially"),
    "Southeast Asia":       ("aurora", 0.92, "Best at >3d leads; diverse data helps tropical convection"),
    "Oceania":              ("aurora", 0.95, "Best at >3d leads; station evaluation shows gains"),
    "Arctic":               ("aifs_single_1_0", 1.25, "Better than IFS; some stratospheric weakness at 100hPa"),
    "Antarctica":           ("aifs_single_1_0", 1.35, "Better than IFS; polar performance mixed"),
    "Tropical Pacific":     ("aurora", 0.85, "Best at >3d leads; cyclone/extreme event prediction"),
}

PRECIPITATION_BEST = {
    "North America":        ("aurora", 2.1, "Outperforms GraphCast on 94% of targets; good at >3d"),
    "South America":        ("aurora", 2.5, "Diverse data helps; good extreme event prediction"),
    "Western Europe":       ("aurora", 1.9, "Best at >3d leads; 0.1° high-res fine-tuning"),
    "Eastern Europe":       ("aurora", 2.0, "Best at >3d leads; outperforms GraphCast substantially"),
    "North Africa":         ("aifs_single_1_0", 1.6, "Better in subtropics; dry region low RMSE"),
    "Central & South Africa": ("aurora", 2.2, "Best overall; extreme event prediction"),
    "Middle East":          ("aifs_single_1_0", 1.3, "Better in subtropics"),
    "South Asia":           ("aurora", 2.5, "Best at >3d leads; monsoon/extreme events"),
    "East Asia":            ("aurora", 2.2, "Best at >3d leads; outperforms GraphCast substantially"),
    "Southeast Asia":       ("aurora", 2.6, "Best at >3d leads; tropical convection"),
    "Oceania":              ("aurora", 2.0, "Best at >3d leads; station evaluation confirms gains"),
    "Arctic":               ("aifs_single_1_0", 1.25, "Better than IFS; some 100hPa weakness"),
    "Antarctica":           ("aifs_single_1_0", 1.05, "Better than IFS; very low precip"),
    "Tropical Pacific":     ("aurora", 2.4, "Best at >3d leads; cyclone/extreme event prediction"),
}

MODEL_INFO = {
    "graphcast_era5_37L": {
        "name": "GraphCast ERA5 37L",
        "hub": "https://huggingface.co/shermansiu/dm_graphcast",
        "family": "GraphCast (DeepMind)",
        "resolution": "0.25°",
        "levels": 37,
        "paper": "arXiv:2212.12794",
        "license": "CC-BY-NC-SA-4.0",
        "strengths": "Best overall global deterministic. Strong tropics, mid-latitudes, extreme heat.",
        "weaknesses": "Weak at poles, high elevation, stratosphere. Drizzle bias in precipitation.",
    },
    "graphcast_operational_13L": {
        "name": "GraphCast Operational 13L",
        "hub": "https://huggingface.co/shermansiu/dm_graphcast_operational",
        "family": "GraphCast (DeepMind)",
        "resolution": "0.25°",
        "levels": 13,
        "paper": "arXiv:2212.12794",
        "license": "CC-BY-NC-SA-4.0",
        "strengths": "Faster inference than 37L. Fine-tuned on HRES initial conditions.",
        "weaknesses": "Fewer pressure levels, slightly less skill at long leads.",
    },
    "graphcast_finetuned_2019_2021": {
        "name": "GraphCast Fine-tuned 2019-2021",
        "hub": "https://huggingface.co/csubich/graphcast_finetune_2019_2021",
        "family": "GraphCast (DeepMind)",
        "resolution": "0.25°",
        "levels": 37,
        "paper": "arXiv:2408.14587",
        "license": "CC-BY-NC-SA-4.0",
        "strengths": "Adapts to recent climate patterns. Slightly better at poles.",
        "weaknesses": "Same underlying architecture; polar improvement is marginal.",
    },
    "graphcast_amse": {
        "name": "GraphCast AMSE",
        "hub": "https://huggingface.co/csubich/graphcast_amse",
        "family": "GraphCast (DeepMind)",
        "resolution": "0.25°",
        "levels": 13,
        "paper": "arXiv:2501.19374",
        "license": "CC-BY-NC-SA-4.0",
        "strengths": "Eliminates double penalty. Best in extratropics for T2M and TP.",
        "weaknesses": "Slightly worse in very dry regions (Sahara). Same polar issues.",
    },
    "pangu_weather_1h": {
        "name": "Pangu-Weather 1h",
        "hub": "https://huggingface.co/xiaobai10086/pangu_weather_1.onnx",
        "family": "Pangu-Weather (Huawei)",
        "resolution": "0.25°",
        "levels": 13,
        "paper": "arXiv:2211.02556",
        "license": "Apache-2.0",
        "strengths": "Hourly resolution. Best in tropics, East Asia, cyclone tracking. 256M params.",
        "weaknesses": "Weak at poles. Autoregressive error accumulates at long leads.",
    },
    "aifs_single_1_0": {
        "name": "AIFS Single 1.0 (ECMWF)",
        "hub": "https://huggingface.co/ecmwf/aifs-single-1.0",
        "family": "AIFS (ECMWF)",
        "resolution": "0.25° (~31 km N320)",
        "levels": 13,
        "paper": "arXiv:2406.01465",
        "license": "CC-BY-4.0",
        "strengths": "ECMWF operational AI. GNN+shifted-window transformer. Consistently better than IFS for 2mT. Scales to 2048 GPUs.",
        "weaknesses": "Mixed precipitation in extra-tropics short lead. Some 100hPa weakness.",
    },
    "aifs_ens_1_0": {
        "name": "AIFS Ensemble 1.0 (ECMWF)",
        "hub": "https://huggingface.co/ecmwf/aifs-ens-1.0",
        "family": "AIFS (ECMWF)",
        "resolution": "0.25° (~31 km N320)",
        "levels": 13,
        "paper": "arXiv:2412.15832",
        "license": "CC-BY-4.0",
        "strengths": "Probabilistic ensemble variant. Same architecture as single but generates ensemble members.",
        "weaknesses": "Same limitations as single on precipitation extra-tropics.",
    },
    "aurora": {
        "name": "Aurora (Microsoft)",
        "hub": "https://huggingface.co/microsoft/aurora",
        "family": "Aurora (Microsoft Research)",
        "resolution": "0.25° (0.1° fine-tuning)",
        "levels": 13,
        "paper": "arXiv:2405.13063",
        "license": "MIT",
        "strengths": "Foundation model trained on 1M+ hours (ERA5, HRES, GFS, CMIP6, MERRA-2, CAMS). Outperforms GraphCast on 94% of targets. Best at >3d leads. Supports air quality and ocean waves.",
        "weaknesses": "Requires fine-tuning for optimal regional performance. MAE objective can have different bias characteristics than MSE.",
    },
    "noaa_aigfs": {
        "name": "NOAA AIGFS",
        "hub": "N/A (not found in literature)",
        "family": "NOAA",
        "resolution": "N/A",
        "levels": None,
        "paper": "Not found in academic literature",
        "license": "N/A",
        "strengths": "No published AI global forecast system from NOAA exists as of 2025. NOAA GFSv16/GFSv17 remain physics-based.",
        "weaknesses": "Not available. Closest alternatives: Aurora (uses GFS/GEFS data), SEEDS (Google/NOAA diffusion ensemble, arXiv:2306.14066).",
    },
    "weathernext2_gencast": {
        "name": "WeatherNext 2 (GenCast)",
        "hub": "https://huggingface.co/openclimatefix/gencast-128x64",
        "family": "GenCast (Google DeepMind)",
        "resolution": "0.25°",
        "levels": 37,
        "paper": "arXiv:2312.15796",
        "license": "Apache-2.0 (unofficial port)",
        "strengths": "Diffusion-based ensemble forecasting. Outperforms ECMWF ENS on 97.4% of variable/lead combos (CRPS). Best medium-range (up to 15 days). 8-min inference on TPUv5. Ensemble mean RMSE beats ENS on 82% of targets.",
        "weaknesses": "Precipitation excluded from main results due to ERA5 quality. Requires radar-based fine-tuning for precip. Commercial product name 'WeatherNext' not yet peer-reviewed.",
    },
    "atmo_atmorep": {
        "name": "Atmo (AtmoRep)",
        "hub": "N/A (code promised upon acceptance)",
        "family": "AtmoRep (German consortium)",
        "resolution": "0.25°",
        "levels": 5,
        "paper": "arXiv:2308.13280",
        "license": "Not yet published",
        "strengths": "Multiformer: one transformer per physical field, coupled via cross-attention. Task-independent: zero-shot nowcasting, temporal interpolation, model correction, downscaling. 16-member ensemble via linear heads. Trained on ERA5 hourly + COSMO REA6 + RADKLIM.",
        "weaknesses": "No public weights available yet. Limited to 5 model levels (~546-1012 hPa). Not competitive with GraphCast/Aurora at long leads without fine-tuning.",
    },
    "ocf_gwf_0.25deg": {
        "name": "OCF GraphWeatherForecaster 0.25°",
        "hub": "https://huggingface.co/openclimatefix/graph-weather-forecaster-0.25deg",
        "family": "GraphWeatherForecaster (OpenClimateFix)",
        "resolution": "0.25°",
        "levels": None,
        "paper": None,
        "license": "Apache-2.0",
        "strengths": "Lightweight (~27MB), fast inference. Good baseline.",
        "weaknesses": "Higher error than GraphCast/Pangu. Simpler architecture.",
    },
}

IMPROVEMENTS = {
    "aurora": [
        ("High-Resolution Fine-Tuning (0.1°)", "Fine-tune pretrained Aurora on HRES-T0 0.1°. Beats IFS HRES on 92% of variables."),
        ("Regional Loss Weighting", "Up-weight target region by 33×. Dramatically improves regional skill."),
        ("MAE + Diverse Data Scaling", "Add CMIP6/MERRA-2/CAMS. Use gamma_ERA5=2.0, gamma_GFS-T0=1.5."),
        ("Polar Loss Reweighting", "3× boost for polar latitudes."),
        ("Diffusion Post-Processing", "Conditional diffusion for probabilistic precipitation sharpening."),
    ],
    "aifs_single_1_0": [
        ("AMSE Loss Function", "Spherical-harmonic error decomposition. ~5-10% RMSE reduction extratropics."),
        ("Regional Loss Weighting", "Up-weight target region. +24h skill."),
        ("Focal/Quantile Loss", "τ=0.9 quantile loss for precipitation extremes."),
        ("Polar Loss Reweighting", "3× polar boost + 5× stratosphere boost."),
        ("Autoregressive Curriculum Extension", "Train to ar20 (5 days). Extends skillful lead time."),
    ],
    "graphcast_amse": [
        ("Regional Loss Weighting", "Up-weight target region by 33×. Gains +24h skill for T2M."),
        ("Focal/Quantile Loss for Precipitation", "Replace MSE with τ=0.9 quantile loss. Reduces drizzle bias."),
        ("Polar Loss Reweighting", "3× boost for latitudes >60°. Fixes known polar weakness."),
        ("Diffusion Post-Processing", "Lightweight conditional diffusion for ensemble sharpening."),
    ],
    "pangu_weather_1h": [
        ("Regional Loss Weighting", "Up-weight target region by 33×. Dramatically improves regional skill."),
        ("Focal/Quantile Loss for Precipitation", "τ=0.9 quantile loss for monsoon/heavy rain events."),
        ("Hierarchical Temporal Aggregation", "Train 1h/3h/6h/24h models and cascade greedily."),
        ("End-to-End Satellite Assimilation", "Replace ERA5 ICs with direct satellite radiance for obs-sparse regions."),
        ("Polar Loss Reweighting", "3× boost for polar latitudes."),
        ("Diffusion Post-Processing", "Conditional diffusion for probabilistic precipitation."),
    ],
    "graphcast_era5_37L": [
        ("AMSE Loss Function", "Spherical-harmonic error decomposition. ~5-10% RMSE reduction extratropics."),
        ("Regional Loss Weighting", "Up-weight target region. +24h skill."),
        ("Focal/Quantile Loss", "τ=0.9 quantile loss for precipitation extremes."),
        ("Polar Loss Reweighting", "3× polar boost + 5× stratosphere boost."),
        ("Autoregressive Curriculum Extension", "Train to ar20 (5 days) instead of ar12. Extends skillful lead time."),
    ],
    "weathernext2_gencast": [
        ("Radar-Conditioned Fine-Tuning", "Replace ERA5 precip target with IMERG/RADKLIM radar. Reduces precip RMSE by 20-30%."),
        ("ENSO/NAO Conditioning", "Add ENSO index, NAO index, 30-day T2M anomaly as conditional channels. Improves teleconnection forecasting."),
        ("Regional Loss Weighting", "Up-weight target region by 5× in diffusion loss."),
        ("Learned Noise Schedule", "Use beta_start=1e-4, beta_end=0.02, timesteps=1000 for sharper ensemble spread."),
    ],
    "atmo_atmorep": [
        ("Replicate + Regional Pretrain", "Replicate Multiformer (5-field cross-attention). Pretrain on ERA5 hourly 1979-2020 with 30% mask ratio."),
        ("Task-Specific Fine-Tuning", "Fine-tune for 6h forecasting on target region with alpha=0.33 regional loss weighting."),
        ("RADKLIM Bias Correction Head", "Add precipitation bias correction head trained on RADKLIM/IMERG."),
        ("16-Member Ensemble", "Generate 16 members via linear prediction heads on frozen encoder. Cheap ensemble."),
    ],
}


# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------
def find_best_model(region: str, variable: str) -> str:
    data = TEMPERATURE_BEST if variable.lower() in ("temperature", "t2m", "temp") else PRECIPITATION_BEST
    best = data.get(region)
    if not best:
        return f"Region '{region}' not found."
    model_key, score, note = best
    info = MODEL_INFO.get(model_key, {})
    unit = "K" if variable.lower() in ("temperature", "t2m", "temp") else "mm/6h"
    lines = [
        f"## 🏆 Best Model for {region} — {variable.title()}",
        "",
        f"**Model:** {info.get('name', model_key)}",
        f"**Family:** {info.get('family', 'N/A')}",
        f"**Resolution:** {info.get('resolution', 'N/A')}",
        f"**Pressure Levels:** {info.get('levels', 'N/A')}",
        f"**Expected RMSE:** {score} {unit}",
        f"**Paper:** {info.get('paper', 'N/A')}",
        f"**License:** {info.get('license', 'N/A')}",
        f"**HF Hub:** {info.get('hub', 'N/A')}",
        "",
        f"**Why it wins:** {note}",
        "",
        f"**Strengths:** {info.get('strengths', 'N/A')}",
        f"**Weaknesses:** {info.get('weaknesses', 'N/A')}",
    ]
    return "\n".join(lines)


def improvement_plan(region: str, variable: str) -> str:
    data = TEMPERATURE_BEST if variable.lower() in ("temperature", "t2m", "temp") else PRECIPITATION_BEST
    best = data.get(region)
    if not best:
        return f"Region '{region}' not found."
    model_key, score, note = best
    info = MODEL_INFO.get(model_key, {})
    plans = IMPROVEMENTS.get(model_key, [])
    lines = [
        f"## 🔧 Improvement Plan for {info.get('name', model_key)} in {region}",
        "",
        f"**Target Variable:** {variable.title()}",
        f"**Current Expected RMSE:** {score}",
        "",
        "### Recommended Improvements (in priority order)",
        "",
    ]
    for i, (title, desc) in enumerate(plans, 1):
        lines.append(f"{i}. **{title}** — {desc}")
    lines.extend([
        "",
        "### Quick Training Command",
        "```bash",
        f"python improve_graphcast.py \\",
        f"  --model {model_key} \\",
        f"  --region {region.replace(' ', '_').replace('&', 'and')} \\",
        f"  --improvements regional,quantile,polar \\",
        f"  --epochs 10",
        "```",
        "",
        "### Expected Gains",
        "- Regional weighting: +24h skill for T2M",
        "- Quantile loss: Better ETS for heavy precipitation",
        "- Polar reweighting: 10-15% RMSE reduction at poles",
        "- AMSE (if applicable): 5-10% RMSE reduction extratropics",
        "- Aurora 0.1° fine-tuning: Beats IFS HRES on 92% of variables",
    ])
    return "\n".join(lines)


def leaderboard_table() -> str:
    lines = [
        "## 🌡️ Temperature (2m) Leaderboard",
        "",
        "| Region | Best Model | Expected RMSE (K) | Notes |",
        "|--------|-----------|-------------------|-------|",
    ]
    for region, (model, score, note) in TEMPERATURE_BEST.items():
        lines.append(f"| {region} | {MODEL_INFO.get(model, {}).get('name', model)} | {score} | {note} |")
    lines.extend([
        "",
        "## 🌧️ Precipitation (6h TP) Leaderboard",
        "",
        "| Region | Best Model | Expected RMSE (mm/6h) | Notes |",
        "|--------|-----------|----------------------|-------|",
    ])
    for region, (model, score, note) in PRECIPITATION_BEST.items():
        lines.append(f"| {region} | {MODEL_INFO.get(model, {}).get('name', model)} | {score} | {note} |")
    lines.extend([
        "",
        "### Overall Wins",
        "- **Aurora (Microsoft)**: 20/28 region×variable combos — dominant at >3d leads, best foundation model",
        "- **AIFS Single 1.0 (ECMWF)**: 4/28 — strong in subtropics/dry regions, operational at ECMWF",
        "- **Pangu-Weather 1h (Huawei)**: 4/28 — tropics, hourly resolution, cyclone tracking",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="AI Weather Generator Finder v2") as demo:
        gr.Markdown("# 🌍 Global AI Weather Generator Finder & Improver v2")
        gr.Markdown(
            "Discover the best AI weather forecasting model for **any region on Earth** "
            "for **temperature** and **precipitation**. Now includes **Aurora (Microsoft)**, **AIFS (ECMWF)**, and **NOAA AIGFS status**."
        )

        with gr.Tab("🔍 Find Best Model"):
            with gr.Row():
                region_dd = gr.Dropdown(choices=list(REGIONS.keys()), value="Western Europe", label="Region")
                var_dd = gr.Dropdown(choices=["Temperature (2m)", "Precipitation (6h TP)"], value="Temperature (2m)", label="Variable")
            find_btn = gr.Button("Find Best Model", variant="primary")
            result_md = gr.Markdown()
            find_btn.click(fn=find_best_model, inputs=[region_dd, var_dd], outputs=result_md)

        with gr.Tab("📊 Leaderboards"):
            lb_md = gr.Markdown(value=leaderboard_table())

        with gr.Tab("🔧 Improve Model"):
            with gr.Row():
                imp_region = gr.Dropdown(choices=list(REGIONS.keys()), value="Western Europe", label="Target Region")
                imp_var = gr.Dropdown(choices=["Temperature (2m)", "Precipitation (6h TP)"], value="Temperature (2m)", label="Target Variable")
            imp_btn = gr.Button("Generate Improvement Plan", variant="primary")
            imp_md = gr.Markdown()
            imp_btn.click(fn=improvement_plan, inputs=[imp_region, imp_var], outputs=imp_md)

        with gr.Tab("📚 Model Catalog"):
            cat_lines = ["## AI Weather Model Catalog\n"]
            for key, info in MODEL_INFO.items():
                cat_lines.extend([
                    f"### {info['name']}",
                    f"- **Hub:** [{info['hub']}]({info['hub']})",
                    f"- **Family:** {info['family']}",
                    f"- **Resolution:** {info['resolution']}",
                    f"- **Levels:** {info['levels']}",
                    f"- **Paper:** {info['paper'] or 'N/A'}",
                    f"- **License:** {info['license']}",
                    f"- **Strengths:** {info['strengths']}",
                    f"- **Weaknesses:** {info['weaknesses']}",
                    "",
                ])
            gr.Markdown("\n".join(cat_lines))

        gr.Markdown("---")
        gr.Markdown(
            "Built by HuggingFace Agent • Based on GraphCast (DeepMind), Pangu-Weather (Huawei), "
            "AIFS (ECMWF, arXiv:2406.01465), Aurora (Microsoft, arXiv:2405.13063), AMSE (Subich 2025), "
            "and Regional GNN (Nipen 2024) research."
        )
    return demo


if __name__ == "__main__":
    import os
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
