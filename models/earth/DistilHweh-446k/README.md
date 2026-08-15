---
license: mit
tags:
- forecast
- weather
- lstm
- classification
- regression
- weather-forecast
- multitask
- harley-ml
- small
---

# Hweh-446k

## Summary

Task: Weather Forecasting  
Inputs: 72 hours time-series  
Outputs: 12h multivariate forecast  
Params: 446k  
Framework: PyTorch

Author: Paul Courneya (Harley-ml)

## Description

**Hweh-446k** is a **446-thousand-parameter LSTM model** distillation of [Hweh-6M](https://huggingface.co/Harley-ml/Hweh-6M) (a ~92% reduction in params!!), trained to predict the next **12 hours of weather**, including temperature, humidity, pressure, precipitation, and more, using the previous **72 hours of weather context**.  
We recommend using this model as a backup to a weather API or for fast offline forecasting when internet access is unavailable.  

We would also like to give a shoutout to [**Open-Meteo**](https://open-meteo.com/) for providing a **free-to-use weather forecasting API**.

### Why “Hweh”?

In Proto-Indo-European, the root ***h₂weh₁-** means “to blow.” We chose it as the name for a weather forecasting model because of its connection to wind and air.

## Architecture

The model uses a multitask LSTM setup:

| Parameter               | Value                                          |
| ----------------------- | ---------------------------------------------- |
| `input_dim`             | `22`                                           |
| `seq_len`               | `72`                                           |
| `num_predict`           | `12`                                           |
| `hidden_dim`            | `128`                                          |
| `num_layers`            | `3`                                            |
| `dropout`               | `0.1`                                          |
| `encoder_type`          | `lstm`                                         |
| `num_locations`         | `82`                                           |
| `location_emb_dim`      | `32`                                           |
| `num_weather_classes`   | `7`                                            |

## Training

We trained Hweh-446k on 4.06 million rows of weather data from 82 locations with the supervision of Hweh-6M for one epoch, using a batch size of 16 and gradient accumulation of 5. Training ran for 4.3 hours on an RTX 2060 6GB GPU.

### Input Features

1. `temperature_2m_norm`
2. `relative_humidity_2m_norm`
3. `apparent_temperature_norm`
4. `precipitation_log_norm`
5. `sea_level_pressure_norm`
6. `surface_pressure_norm`
7. `cloud_cover_total_norm`
8. `visibility_norm`
9. `wind_speed_10m_norm`
10. `wind_direction_10m_sin`
11. `wind_direction_10m_cos`
12. `hour_sin`
13. `hour_cos`
14. `day_of_year_sin`
15. `day_of_year_cos`
16. `weather_code_onehot_clear`
17. `weather_code_onehot_cloudy`
18. `weather_code_onehot_fog`
19. `weather_code_onehot_drizzle`
20. `weather_code_onehot_rain`
21. `weather_code_onehot_snow`
22. `weather_code_onehot_thunderstorm`

### Output Features

1. `y_temp_c`: continuous regression
2. `y_humidity`: continuous regression
3. `y_apparent_temperature`: continuous regression
4. `y_precipitation_mm`: continuous regression
5. `y_sea_level_pressure_hpa`: continuous regression
6. `y_surface_pressure_hpa`: continuous regression
7. `y_cloud_cover_total`: continuous regression
8. `y_wind_speed_10m`: continuous regression
9. `y_wind_direction_sin`: continuous regression
10. `y_wind_direction_cos`: continuous regression
11. `y_rain_prob`: binary classification
12. `y_weather_class`: multiclass classification

### Training Results

#### Training & Evaluation Metrics

|  Step | Train Loss | Eval Loss | Weather Acc | Rain Acc | Rain Recall | Weather Recall |
| ----: | ---------: | --------: | ----------: | -------: | ----------: | -------------: |
|    1k |     8.4609 |    8.8471 |      0.6317 |   0.7451 |      0.7640 |         0.2574 |
|    5k |     5.1420 |    5.0602 |      0.6247 |   0.7531 |      0.8025 |         0.5648 |
|   10k |     4.1733 |    3.9198 |      0.6117 |   0.7876 |      0.8016 |         0.6297 |
|   15k |     3.8354 |    3.6310 |      0.6140 |   0.7920 |      0.8009 |         0.6187 |
|   20k |     3.6206 |    3.4365 |      0.6083 |   0.7881 |      0.8140 |         0.6179 |
|   25k |     3.5378 |    3.3251 |      0.6083 |   0.7859 |      0.8173 |         0.6245 |
|   30k |     3.4534 |    3.2846 |      0.6041 |   0.7812 |      0.8272 |         0.6398 |
|   35k |     3.4272 |    3.2324 |      0.6061 |   0.7860 |      0.8194 |         0.6289 |
|   40k |     3.4143 |    3.2230 |      0.6080 |   0.7862 |      0.8200 |         0.6339 |
| 42.6k |          — |    3.2180 |      0.6081 |   0.7857 |      0.8212 |         0.6340 |

Note: Loss looks higher than Hweh-6M's because of KL + train/val loss.
#### Regression Error Metrics (MAE)

|  Step | Apparent |   Cloud | Humidity | Precip (mm) | Sea Level P | Surface P |   Temp |  Wind |
| ----: | -------: | ------: | -------: | ----------: | ----------: | --------: | -----: | ----: |
|    1k |   212.80 | 2179.70 |  1476.42 |       0.140 |     7571.45 |  83590.98 | 172.79 | 60.49 |
|    5k |     2.28 |   25.58 |     9.04 |       0.107 |        3.50 |     14.55 |   1.90 |  3.78 |
|   10k |     2.06 |   25.31 |     8.08 |       0.100 |        3.31 |      9.63 |   1.72 |  3.37 |
|   15k |     1.91 |   25.00 |     7.88 |       0.101 |        3.18 |      7.93 |   1.61 |  3.25 |
|   20k |     1.88 |   25.12 |     7.60 |       0.101 |        3.13 |      7.41 |   1.56 |  3.18 |
|   25k |     1.84 |   25.01 |     7.53 |       0.102 |        3.09 |      6.61 |   1.53 |  3.13 |
|   30k |     1.81 |   25.03 |     7.45 |       0.102 |        3.12 |      6.60 |   1.51 |  3.12 |
|   35k |     1.81 |   24.94 |     7.42 |       0.101 |        3.07 |      6.39 |   1.52 |  3.12 |
|   40k |     1.79 |   24.94 |     7.39 |       0.101 |        3.06 |      6.37 |   1.50 |  3.11 |
| 42.6k |     1.79 |   24.92 |     7.39 |       0.101 |        3.06 |      6.38 |   1.50 |  3.11 |

This model did better than the teacher on MAE and accuracy, but the real-world accuracy is 5-10% worse. 

## Generation Examples

| ID | Class        |
| -- | ------------ |
| 0  | clear        |
| 1  | cloudy       |
| 2  | fog          |
| 3  | drizzle      |
| 4  | rain         |
| 5  | snow         |
| 6  | thunderstorm |

City=Seattle
```
{
  "city": "Seattle",
  "location_id": "1",
  "model_location_id": 0,
  "data_source": "open-meteo forecast api (past-hours context only)",
  "requested_at_utc": "2026-05-08T19:57:14.429521+00:00",
  "context": {
    "hours": 72,
    "start_utc": "2026-05-05T19:00:00+00:00",
    "end_utc": "2026-05-08T18:00:00+00:00",
    "start_local": "2026-05-05T12:00:00-07:00",
    "end_local": "2026-05-08T11:00:00-07:00"
  },
  "model": {
    "encoder_type": "lstm",
    "seq_len": 72,
    "input_dim": 22,
    "num_weather_classes": 7
  },
  "forecast": [
    {
      "lead_hours": 1,
      "target_utc": "2026-05-08T19:00:00+00:00",
      "target_local": "2026-05-08T12:00:00-07:00",
      "temperature_2m_c": 12.21396255493164,
      "relative_humidity_2m_pct": 72.33454895019531,
      "apparent_temperature_c": 10.097986221313477,
      "precipitation_mm": 0.015628309920430183,
      "pressure_msl_hpa": 1022.0569458007812,
      "surface_pressure_hpa": 1014.205078125,
      "cloud_cover_pct": 94.34225463867188,
      "wind_speed_10m_kmh": 12.568346977233887,
      "rain_probability": 0.19356799125671387,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.020174680277705193,
        "class_1": 0.9282320737838745,
        "class_2": 0.0022441258188337088,
        "class_3": 0.04022064805030823,
        "class_4": 0.008552632294595242,
        "class_5": 0.0005501594278030097,
        "class_6": 2.556406798248645e-05
      }
    },
    {
      "lead_hours": 2,
      "target_utc": "2026-05-08T20:00:00+00:00",
      "target_local": "2026-05-08T13:00:00-07:00",
      "temperature_2m_c": 12.8738374710083,
      "relative_humidity_2m_pct": 70.51017761230469,
      "apparent_temperature_c": 10.80291748046875,
      "precipitation_mm": 0.011432276107370853,
      "pressure_msl_hpa": 1022.0043334960938,
      "surface_pressure_hpa": 1014.2881469726562,
      "cloud_cover_pct": 89.5630111694336,
      "wind_speed_10m_kmh": 12.822803497314453,
      "rain_probability": 0.2689012587070465,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.04770936816930771,
        "class_1": 0.8698592185974121,
        "class_2": 0.0019157826900482178,
        "class_3": 0.057819921523332596,
        "class_4": 0.02183685451745987,
        "class_5": 0.0008237561560235918,
        "class_6": 3.5158725950168446e-05
      }
    },
    {
      "lead_hours": 3,
      "target_utc": "2026-05-08T21:00:00+00:00",
      "target_local": "2026-05-08T14:00:00-07:00",
      "temperature_2m_c": 13.51952075958252,
      "relative_humidity_2m_pct": 68.44591522216797,
      "apparent_temperature_c": 11.500271797180176,
      "precipitation_mm": 0.006943895947188139,
      "pressure_msl_hpa": 1021.80859375,
      "surface_pressure_hpa": 1014.2529296875,
      "cloud_cover_pct": 84.49480438232422,
      "wind_speed_10m_kmh": 12.941960334777832,
      "rain_probability": 0.30342426896095276,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.07170139998197556,
        "class_1": 0.8286910057067871,
        "class_2": 0.0013093570014461875,
        "class_3": 0.06433302909135818,
        "class_4": 0.03300558775663376,
        "class_5": 0.0009036734118126333,
        "class_6": 5.590655928244814e-05
      }
    },
    {
      "lead_hours": 4,
      "target_utc": "2026-05-08T22:00:00+00:00",
      "target_local": "2026-05-08T15:00:00-07:00",
      "temperature_2m_c": 13.970871925354004,
      "relative_humidity_2m_pct": 66.8187026977539,
      "apparent_temperature_c": 11.959537506103516,
      "precipitation_mm": 0.009790810756385326,
      "pressure_msl_hpa": 1021.4691162109375,
      "surface_pressure_hpa": 1014.1052856445312,
      "cloud_cover_pct": 80.34271240234375,
      "wind_speed_10m_kmh": 13.050889015197754,
      "rain_probability": 0.33110707998275757,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.10919982939958572,
        "class_1": 0.7759758830070496,
        "class_2": 0.0011831748997792602,
        "class_3": 0.07015678286552429,
        "class_4": 0.042580746114254,
        "class_5": 0.000857028178870678,
        "class_6": 4.6529316023224965e-05
      }
    },
    {
      "lead_hours": 5,
      "target_utc": "2026-05-08T23:00:00+00:00",
      "target_local": "2026-05-08T16:00:00-07:00",
      "temperature_2m_c": 14.132287979125977,
      "relative_humidity_2m_pct": 66.1208267211914,
      "apparent_temperature_c": 12.156023025512695,
      "precipitation_mm": 0.008600466884672642,
      "pressure_msl_hpa": 1021.0518188476562,
      "surface_pressure_hpa": 1013.7891845703125,
      "cloud_cover_pct": 76.06925201416016,
      "wind_speed_10m_kmh": 12.926268577575684,
      "rain_probability": 0.3409281373023987,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.1305426061153412,
        "class_1": 0.7395681142807007,
        "class_2": 0.0007046378450468183,
        "class_3": 0.07573042809963226,
        "class_4": 0.052331216633319855,
        "class_5": 0.0010767169296741486,
        "class_6": 4.629969771485776e-05
      }
    },
    {
      "lead_hours": 6,
      "target_utc": "2026-05-09T00:00:00+00:00",
      "target_local": "2026-05-08T17:00:00-07:00",
      "temperature_2m_c": 13.963343620300293,
      "relative_humidity_2m_pct": 66.67638397216797,
      "apparent_temperature_c": 11.971813201904297,
      "precipitation_mm": 0.010375693440437317,
      "pressure_msl_hpa": 1020.6505126953125,
      "surface_pressure_hpa": 1013.4520874023438,
      "cloud_cover_pct": 73.21341705322266,
      "wind_speed_10m_kmh": 12.721481323242188,
      "rain_probability": 0.35606324672698975,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.15235546231269836,
        "class_1": 0.7133304476737976,
        "class_2": 0.0007980632944963872,
        "class_3": 0.07519367337226868,
        "class_4": 0.05724283307790756,
        "class_5": 0.0010369947412982583,
        "class_6": 4.249428093316965e-05
      }
    },
    {
      "lead_hours": 7,
      "target_utc": "2026-05-09T01:00:00+00:00",
      "target_local": "2026-05-08T18:00:00-07:00",
      "temperature_2m_c": 13.449448585510254,
      "relative_humidity_2m_pct": 68.29602813720703,
      "apparent_temperature_c": 11.426795959472656,
      "precipitation_mm": 0.012202607467770576,
      "pressure_msl_hpa": 1020.3434448242188,
      "surface_pressure_hpa": 1013.139892578125,
      "cloud_cover_pct": 70.92017364501953,
      "wind_speed_10m_kmh": 12.359071731567383,
      "rain_probability": 0.3651714026927948,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.16448773443698883,
        "class_1": 0.695494532585144,
        "class_2": 0.0008648976217955351,
        "class_3": 0.07075871527194977,
        "class_4": 0.06722015887498856,
        "class_5": 0.0011408632853999734,
        "class_6": 3.3135267585748807e-05
      }
    },
    {
      "lead_hours": 8,
      "target_utc": "2026-05-09T02:00:00+00:00",
      "target_local": "2026-05-08T19:00:00-07:00",
      "temperature_2m_c": 12.755823135375977,
      "relative_humidity_2m_pct": 70.44146728515625,
      "apparent_temperature_c": 10.662925720214844,
      "precipitation_mm": 0.014662384055554867,
      "pressure_msl_hpa": 1020.1875610351562,
      "surface_pressure_hpa": 1012.8895874023438,
      "cloud_cover_pct": 69.15129852294922,
      "wind_speed_10m_kmh": 11.787208557128906,
      "rain_probability": 0.3489035665988922,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.19570188224315643,
        "class_1": 0.6735588908195496,
        "class_2": 0.000978420372121036,
        "class_3": 0.06184739992022514,
        "class_4": 0.0664038360118866,
        "class_5": 0.0014616982080042362,
        "class_6": 4.789666854776442e-05
      }
    },
    {
      "lead_hours": 9,
      "target_utc": "2026-05-09T03:00:00+00:00",
      "target_local": "2026-05-08T20:00:00-07:00",
      "temperature_2m_c": 11.955390930175781,
      "relative_humidity_2m_pct": 73.05667877197266,
      "apparent_temperature_c": 9.77428913116455,
      "precipitation_mm": 0.015376528725028038,
      "pressure_msl_hpa": 1020.2242431640625,
      "surface_pressure_hpa": 1012.7984619140625,
      "cloud_cover_pct": 67.46344757080078,
      "wind_speed_10m_kmh": 11.127586364746094,
      "rain_probability": 0.36496502161026,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.20349998772144318,
        "class_1": 0.6480608582496643,
        "class_2": 0.0010103220120072365,
        "class_3": 0.06373731791973114,
        "class_4": 0.08206527680158615,
        "class_5": 0.0015911461086943746,
        "class_6": 3.510143869789317e-05
      }
    },
    {
      "lead_hours": 10,
      "target_utc": "2026-05-09T04:00:00+00:00",
      "target_local": "2026-05-08T21:00:00-07:00",
      "temperature_2m_c": 11.182319641113281,
      "relative_humidity_2m_pct": 75.5196304321289,
      "apparent_temperature_c": 8.978246688842773,
      "precipitation_mm": 0.016823438927531242,
      "pressure_msl_hpa": 1020.421875,
      "surface_pressure_hpa": 1012.8126831054688,
      "cloud_cover_pct": 65.73115539550781,
      "wind_speed_10m_kmh": 10.359850883483887,
      "rain_probability": 0.35924479365348816,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.21845510601997375,
        "class_1": 0.6389062404632568,
        "class_2": 0.0018773162737488747,
        "class_3": 0.059891991317272186,
        "class_4": 0.07879848033189774,
        "class_5": 0.002034474164247513,
        "class_6": 3.633901724242605e-05
      }
    },
    {
      "lead_hours": 11,
      "target_utc": "2026-05-09T05:00:00+00:00",
      "target_local": "2026-05-08T22:00:00-07:00",
      "temperature_2m_c": 10.499757766723633,
      "relative_humidity_2m_pct": 77.536865234375,
      "apparent_temperature_c": 8.306406021118164,
      "precipitation_mm": 0.01860857754945755,
      "pressure_msl_hpa": 1020.7318725585938,
      "surface_pressure_hpa": 1012.9368896484375,
      "cloud_cover_pct": 65.10720825195312,
      "wind_speed_10m_kmh": 9.62015151977539,
      "rain_probability": 0.3694237470626831,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.22774139046669006,
        "class_1": 0.6242526173591614,
        "class_2": 0.0028147574048489332,
        "class_3": 0.06025313213467598,
        "class_4": 0.0822005346417427,
        "class_5": 0.0026846849359571934,
        "class_6": 5.289047476253472e-05
      }
    },
    {
      "lead_hours": 12,
      "target_utc": "2026-05-09T06:00:00+00:00",
      "target_local": "2026-05-08T23:00:00-07:00",
      "temperature_2m_c": 9.956731796264648,
      "relative_humidity_2m_pct": 79.21904754638672,
      "apparent_temperature_c": 7.87125301361084,
      "precipitation_mm": 0.017959173768758774,
      "pressure_msl_hpa": 1021.0579833984375,
      "surface_pressure_hpa": 1013.093994140625,
      "cloud_cover_pct": 64.14817810058594,
      "wind_speed_10m_kmh": 8.923616409301758,
      "rain_probability": 0.3691202700138092,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.23541118204593658,
        "class_1": 0.618242621421814,
        "class_2": 0.0044443667866289616,
        "class_3": 0.06440076231956482,
        "class_4": 0.0741729885339737,
        "class_5": 0.003270090091973543,
        "class_6": 5.8019890275318176e-05
      }
    }
  ],
  "sanity": {
    "sequence_shape": [
      72,
      22
    ],
    "finite_features": true
  }
}
PS C:\Users\Paulc> python3.12 weather_infer.py --model_dir "C:\Users\Paulc\weather_model\student_distilled" --city Seattle
Warning: unexpected keys while loading checkpoint: ['distill_proj.weight']
{
  "city": "Seattle",
  "location_id": "1",
  "model_location_id": 0,
  "data_source": "open-meteo forecast api (past-hours context only)",
  "requested_at_utc": "2026-05-08T19:57:47.439276+00:00",
  "context": {
    "hours": 72,
    "start_utc": "2026-05-05T19:00:00+00:00",
    "end_utc": "2026-05-08T18:00:00+00:00",
    "start_local": "2026-05-05T12:00:00-07:00",
    "end_local": "2026-05-08T11:00:00-07:00"
  },
  "model": {
    "encoder_type": "lstm",
    "seq_len": 72,
    "input_dim": 22,
    "num_weather_classes": 7
  },
  "forecast": [
    {
      "lead_hours": 1,
      "target_utc": "2026-05-08T19:00:00+00:00",
      "target_local": "2026-05-08T12:00:00-07:00",
      "temperature_2m_c": 13.681238174438477,
      "relative_humidity_2m_pct": 69.90876770019531,
      "apparent_temperature_c": 11.687149047851562,
      "precipitation_mm": 0.0012515264097601175,
      "pressure_msl_hpa": 1019.3030395507812,
      "surface_pressure_hpa": 1018.5359497070312,
      "cloud_cover_pct": 88.76920318603516,
      "wind_speed_10m_kmh": 13.126839637756348,
      "rain_probability": 0.1457637995481491,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.02097843959927559,
        "class_1": 0.9304905533790588,
        "class_2": 0.0025352241937071085,
        "class_3": 0.03495039418339729,
        "class_4": 0.01054275780916214,
        "class_5": 0.0004654920194298029,
        "class_6": 3.7044908822281286e-05
      }
    },
    {
      "lead_hours": 2,
      "target_utc": "2026-05-08T20:00:00+00:00",
      "target_local": "2026-05-08T13:00:00-07:00",
      "temperature_2m_c": 14.486506462097168,
      "relative_humidity_2m_pct": 67.52698516845703,
      "apparent_temperature_c": 12.55270767211914,
      "precipitation_mm": 0.0008608415955677629,
      "pressure_msl_hpa": 1019.0198364257812,
      "surface_pressure_hpa": 1018.4589233398438,
      "cloud_cover_pct": 84.31889343261719,
      "wind_speed_10m_kmh": 13.241435050964355,
      "rain_probability": 0.19527363777160645,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.053049832582473755,
        "class_1": 0.8794819712638855,
        "class_2": 0.0021074640098959208,
        "class_3": 0.0436088852584362,
        "class_4": 0.02111213095486164,
        "class_5": 0.0005800225771963596,
        "class_6": 5.971627979306504e-05
      }
    },
    {
      "lead_hours": 3,
      "target_utc": "2026-05-08T21:00:00+00:00",
      "target_local": "2026-05-08T14:00:00-07:00",
      "temperature_2m_c": 15.089279174804688,
      "relative_humidity_2m_pct": 65.6773681640625,
      "apparent_temperature_c": 13.181319236755371,
      "precipitation_mm": 0.002190209459513426,
      "pressure_msl_hpa": 1018.709716796875,
      "surface_pressure_hpa": 1018.2867431640625,
      "cloud_cover_pct": 80.14619445800781,
      "wind_speed_10m_kmh": 13.32516860961914,
      "rain_probability": 0.21867528557777405,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.08254168927669525,
        "class_1": 0.8380688428878784,
        "class_2": 0.0014072611229494214,
        "class_3": 0.04709410294890404,
        "class_4": 0.030200183391571045,
        "class_5": 0.0006088378722779453,
        "class_6": 7.912428554845974e-05
      }
    },
    {
      "lead_hours": 4,
      "target_utc": "2026-05-08T22:00:00+00:00",
      "target_local": "2026-05-08T15:00:00-07:00",
      "temperature_2m_c": 15.403922080993652,
      "relative_humidity_2m_pct": 64.67561340332031,
      "apparent_temperature_c": 13.48654556274414,
      "precipitation_mm": 0.0021571130491793156,
      "pressure_msl_hpa": 1018.3944702148438,
      "surface_pressure_hpa": 1018.0625,
      "cloud_cover_pct": 76.4104995727539,
      "wind_speed_10m_kmh": 13.275524139404297,
      "rain_probability": 0.2299734503030777,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.11604393273591995,
        "class_1": 0.7922014594078064,
        "class_2": 0.0011422870447859168,
        "class_3": 0.05158008262515068,
        "class_4": 0.03833876550197601,
        "class_5": 0.000621610670350492,
        "class_6": 7.179530075518414e-05
      }
    },
    {
      "lead_hours": 5,
      "target_utc": "2026-05-08T23:00:00+00:00",
      "target_local": "2026-05-08T16:00:00-07:00",
      "temperature_2m_c": 15.407997131347656,
      "relative_humidity_2m_pct": 64.65668487548828,
      "apparent_temperature_c": 13.48292350769043,
      "precipitation_mm": 0.0026813943404704332,
      "pressure_msl_hpa": 1018.1220703125,
      "surface_pressure_hpa": 1017.7960205078125,
      "cloud_cover_pct": 72.77561950683594,
      "wind_speed_10m_kmh": 13.133893966674805,
      "rain_probability": 0.23628145456314087,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.14982207119464874,
        "class_1": 0.7523030638694763,
        "class_2": 0.0008282885537482798,
        "class_3": 0.05172016844153404,
        "class_4": 0.04454538971185684,
        "class_5": 0.0007084720418788493,
        "class_6": 7.257604011101648e-05
      }
    },
    {
      "lead_hours": 6,
      "target_utc": "2026-05-09T00:00:00+00:00",
      "target_local": "2026-05-08T17:00:00-07:00",
      "temperature_2m_c": 15.139252662658691,
      "relative_humidity_2m_pct": 65.5518798828125,
      "apparent_temperature_c": 13.192585945129395,
      "precipitation_mm": 0.0026606114115566015,
      "pressure_msl_hpa": 1017.9351196289062,
      "surface_pressure_hpa": 1017.5604858398438,
      "cloud_cover_pct": 69.60166931152344,
      "wind_speed_10m_kmh": 12.789706230163574,
      "rain_probability": 0.23617199063301086,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.17920561134815216,
        "class_1": 0.7213184237480164,
        "class_2": 0.0008871846948750317,
        "class_3": 0.05144878104329109,
        "class_4": 0.046307649463415146,
        "class_5": 0.0007639037212356925,
        "class_6": 6.843609298812225e-05
      }
    },
    {
      "lead_hours": 7,
      "target_utc": "2026-05-09T01:00:00+00:00",
      "target_local": "2026-05-08T18:00:00-07:00",
      "temperature_2m_c": 14.66390609741211,
      "relative_humidity_2m_pct": 67.1559066772461,
      "apparent_temperature_c": 12.692171096801758,
      "precipitation_mm": 0.002722225384786725,
      "pressure_msl_hpa": 1017.8424072265625,
      "surface_pressure_hpa": 1017.3685913085938,
      "cloud_cover_pct": 66.97268676757812,
      "wind_speed_10m_kmh": 12.329818725585938,
      "rain_probability": 0.23410728573799133,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.20177967846393585,
        "class_1": 0.697116494178772,
        "class_2": 0.001064434414729476,
        "class_3": 0.04921679198741913,
        "class_4": 0.049886710941791534,
        "class_5": 0.0008780939970165491,
        "class_6": 5.7844863476930186e-05
      }
    },
    {
      "lead_hours": 8,
      "target_utc": "2026-05-09T02:00:00+00:00",
      "target_local": "2026-05-08T19:00:00-07:00",
      "temperature_2m_c": 14.042488098144531,
      "relative_humidity_2m_pct": 69.15681457519531,
      "apparent_temperature_c": 12.045327186584473,
      "precipitation_mm": 0.003981542307883501,
      "pressure_msl_hpa": 1017.853759765625,
      "surface_pressure_hpa": 1017.2957763671875,
      "cloud_cover_pct": 64.85920715332031,
      "wind_speed_10m_kmh": 11.780016899108887,
      "rain_probability": 0.23837216198444366,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.23561197519302368,
        "class_1": 0.6629505753517151,
        "class_2": 0.0012489539803937078,
        "class_3": 0.04817439988255501,
        "class_4": 0.050942711532115936,
        "class_5": 0.0010061608627438545,
        "class_6": 6.52461385470815e-05
      }
    },
    {
      "lead_hours": 9,
      "target_utc": "2026-05-09T03:00:00+00:00",
      "target_local": "2026-05-08T20:00:00-07:00",
      "temperature_2m_c": 13.325971603393555,
      "relative_humidity_2m_pct": 71.42361450195312,
      "apparent_temperature_c": 11.300540924072266,
      "precipitation_mm": 0.0030152045655995607,
      "pressure_msl_hpa": 1017.9505004882812,
      "surface_pressure_hpa": 1017.2774047851562,
      "cloud_cover_pct": 63.037109375,
      "wind_speed_10m_kmh": 11.165238380432129,
      "rain_probability": 0.23051044344902039,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.24986012279987335,
        "class_1": 0.6375465989112854,
        "class_2": 0.0016154011245816946,
        "class_3": 0.04874761775135994,
        "class_4": 0.06096767634153366,
        "class_5": 0.0012142135528847575,
        "class_6": 4.836557855014689e-05
      }
    },
    {
      "lead_hours": 10,
      "target_utc": "2026-05-09T04:00:00+00:00",
      "target_local": "2026-05-08T21:00:00-07:00",
      "temperature_2m_c": 12.574642181396484,
      "relative_humidity_2m_pct": 73.7841796875,
      "apparent_temperature_c": 10.549814224243164,
      "precipitation_mm": 0.004971037618815899,
      "pressure_msl_hpa": 1018.10400390625,
      "surface_pressure_hpa": 1017.2828979492188,
      "cloud_cover_pct": 61.4162483215332,
      "wind_speed_10m_kmh": 10.5538911819458,
      "rain_probability": 0.23788221180438995,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.27152255177497864,
        "class_1": 0.6182448863983154,
        "class_2": 0.0025821358431130648,
        "class_3": 0.04885515570640564,
        "class_4": 0.05706281587481499,
        "class_5": 0.0016854261048138142,
        "class_6": 4.704758248408325e-05
      }
    },
    {
      "lead_hours": 11,
      "target_utc": "2026-05-09T05:00:00+00:00",
      "target_local": "2026-05-08T22:00:00-07:00",
      "temperature_2m_c": 11.85836124420166,
      "relative_humidity_2m_pct": 75.99488830566406,
      "apparent_temperature_c": 9.845465660095215,
      "precipitation_mm": 0.0059099141508340836,
      "pressure_msl_hpa": 1018.2722778320312,
      "surface_pressure_hpa": 1017.3274536132812,
      "cloud_cover_pct": 60.944053649902344,
      "wind_speed_10m_kmh": 10.019789695739746,
      "rain_probability": 0.24793456494808197,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.27306604385375977,
        "class_1": 0.6107510328292847,
        "class_2": 0.004449337720870972,
        "class_3": 0.0498417466878891,
        "class_4": 0.05973493307828903,
        "class_5": 0.002092213137075305,
        "class_6": 6.465442857006565e-05
      }
    },
    {
      "lead_hours": 12,
      "target_utc": "2026-05-09T06:00:00+00:00",
      "target_local": "2026-05-08T23:00:00-07:00",
      "temperature_2m_c": 11.196554183959961,
      "relative_humidity_2m_pct": 78.09349060058594,
      "apparent_temperature_c": 9.231935501098633,
      "precipitation_mm": 0.00701399240642786,
      "pressure_msl_hpa": 1018.4185791015625,
      "surface_pressure_hpa": 1017.3386840820312,
      "cloud_cover_pct": 60.26295471191406,
      "wind_speed_10m_kmh": 9.483912467956543,
      "rain_probability": 0.2565533518791199,
      "weather_class": 1,
      "weather_class_name": "class_1",
      "weather_class_probabilities": {
        "class_0": 0.27720507979393005,
        "class_1": 0.6052833795547485,
        "class_2": 0.007175811566412449,
        "class_3": 0.04983551800251007,
        "class_4": 0.05781771242618561,
        "class_5": 0.0026247103232890368,
        "class_6": 5.782112566521391e-05
      }
    }
  ],
  "sanity": {
    "sequence_shape": [
      72,
      22
    ],
    "finite_features": true
  }
}
```

City=Nuuk
```
{
  "city": "Nuuk",
  "location_id": "83",
  "model_location_id": 0,
  "data_source": "open-meteo forecast api (past-hours context only)",
  "requested_at_utc": "2026-05-08T20:40:35.109779+00:00",
  "context": {
    "hours": 72,
    "start_utc": "2026-05-05T20:00:00+00:00",
    "end_utc": "2026-05-08T19:00:00+00:00",
    "start_local": "2026-05-05T19:00:00-01:00",
    "end_local": "2026-05-08T18:00:00-01:00"
  },
  "model": {
    "encoder_type": "lstm",
    "seq_len": 72,
    "input_dim": 22,
    "num_weather_classes": 7
  },
  "forecast": [
    {
      "lead_hours": 1,
      "target_utc": "2026-05-08T20:00:00+00:00",
      "target_local": "2026-05-08T19:00:00-01:00",
      "temperature_2m_c": 5.2753753662109375,
      "relative_humidity_2m_pct": 93.01068115234375,
      "apparent_temperature_c": 1.6396684646606445,
      "precipitation_mm": 0.3556472063064575,
      "pressure_msl_hpa": 1005.5432739257812,
      "surface_pressure_hpa": 973.415771484375,
      "cloud_cover_pct": 98.54638671875,
      "wind_speed_10m_kmh": 13.717008590698242,
      "rain_probability": 0.9789170026779175,
      "weather_class": 3,
      "weather_class_name": "class_3",
      "weather_class_probabilities": {
        "class_0": 0.000619232130702585,
        "class_1": 0.057769663631916046,
        "class_2": 0.003395488252863288,
        "class_3": 0.401492714881897,
        "class_4": 0.18206973373889923,
        "class_5": 0.3545896112918854,
        "class_6": 6.364739965647459e-05
      }
    },
    {
      "lead_hours": 2,
      "target_utc": "2026-05-08T21:00:00+00:00",
      "target_local": "2026-05-08T20:00:00-01:00",
      "temperature_2m_c": 4.986588478088379,
      "relative_humidity_2m_pct": 93.84243774414062,
      "apparent_temperature_c": 1.352757453918457,
      "precipitation_mm": 0.2776345908641815,
      "pressure_msl_hpa": 1005.599853515625,
      "surface_pressure_hpa": 973.536376953125,
      "cloud_cover_pct": 98.45586395263672,
      "wind_speed_10m_kmh": 13.445389747619629,
      "rain_probability": 0.9556259512901306,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.00150326790753752,
        "class_1": 0.09517476707696915,
        "class_2": 0.004558710381388664,
        "class_3": 0.32409900426864624,
        "class_4": 0.1846529245376587,
        "class_5": 0.38996437191963196,
        "class_6": 4.702423757407814e-05
      }
    },
    {
      "lead_hours": 3,
      "target_utc": "2026-05-08T22:00:00+00:00",
      "target_local": "2026-05-08T21:00:00-01:00",
      "temperature_2m_c": 4.788308143615723,
      "relative_humidity_2m_pct": 94.0885238647461,
      "apparent_temperature_c": 1.1667909622192383,
      "precipitation_mm": 0.23039932548999786,
      "pressure_msl_hpa": 1005.703857421875,
      "surface_pressure_hpa": 973.5965576171875,
      "cloud_cover_pct": 97.65797424316406,
      "wind_speed_10m_kmh": 13.209211349487305,
      "rain_probability": 0.9299042820930481,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.002542331349104643,
        "class_1": 0.12809209525585175,
        "class_2": 0.0058285691775381565,
        "class_3": 0.3043138086795807,
        "class_4": 0.17225369811058044,
        "class_5": 0.3869440257549286,
        "class_6": 2.5515650122542866e-05
      }
    },
    {
      "lead_hours": 4,
      "target_utc": "2026-05-08T23:00:00+00:00",
      "target_local": "2026-05-08T22:00:00-01:00",
      "temperature_2m_c": 4.660131454467773,
      "relative_humidity_2m_pct": 94.03594970703125,
      "apparent_temperature_c": 1.0469179153442383,
      "precipitation_mm": 0.19706439971923828,
      "pressure_msl_hpa": 1005.7493896484375,
      "surface_pressure_hpa": 973.7083740234375,
      "cloud_cover_pct": 97.17306518554688,
      "wind_speed_10m_kmh": 13.017566680908203,
      "rain_probability": 0.9050359129905701,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.004407276399433613,
        "class_1": 0.15539932250976562,
        "class_2": 0.009657401591539383,
        "class_3": 0.2794102430343628,
        "class_4": 0.1521354615688324,
        "class_5": 0.3989632725715637,
        "class_6": 2.7043566660722718e-05
      }
    },
    {
      "lead_hours": 5,
      "target_utc": "2026-05-09T00:00:00+00:00",
      "target_local": "2026-05-08T23:00:00-01:00",
      "temperature_2m_c": 4.5112457275390625,
      "relative_humidity_2m_pct": 93.88682556152344,
      "apparent_temperature_c": 0.9274702072143555,
      "precipitation_mm": 0.1685791015625,
      "pressure_msl_hpa": 1005.7725830078125,
      "surface_pressure_hpa": 973.7322387695312,
      "cloud_cover_pct": 96.03288269042969,
      "wind_speed_10m_kmh": 12.944330215454102,
      "rain_probability": 0.8804075121879578,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.006306177470833063,
        "class_1": 0.17262385785579681,
        "class_2": 0.00996350683271885,
        "class_3": 0.2658991515636444,
        "class_4": 0.1401163786649704,
        "class_5": 0.4050598442554474,
        "class_6": 3.1046529329614714e-05
      }
    },
    {
      "lead_hours": 6,
      "target_utc": "2026-05-09T01:00:00+00:00",
      "target_local": "2026-05-09T00:00:00-01:00",
      "temperature_2m_c": 4.33610725402832,
      "relative_humidity_2m_pct": 94.00520324707031,
      "apparent_temperature_c": 0.7778654098510742,
      "precipitation_mm": 0.14649224281311035,
      "pressure_msl_hpa": 1005.8167114257812,
      "surface_pressure_hpa": 973.7780151367188,
      "cloud_cover_pct": 95.56141662597656,
      "wind_speed_10m_kmh": 12.845012664794922,
      "rain_probability": 0.8599434494972229,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.007768102455884218,
        "class_1": 0.18894362449645996,
        "class_2": 0.011767406016588211,
        "class_3": 0.2329735904932022,
        "class_4": 0.12322919070720673,
        "class_5": 0.43529438972473145,
        "class_6": 2.3752647393848747e-05
      }
    },
    {
      "lead_hours": 7,
      "target_utc": "2026-05-09T02:00:00+00:00",
      "target_local": "2026-05-09T01:00:00-01:00",
      "temperature_2m_c": 4.140122413635254,
      "relative_humidity_2m_pct": 94.25415802001953,
      "apparent_temperature_c": 0.5866508483886719,
      "precipitation_mm": 0.13218756020069122,
      "pressure_msl_hpa": 1005.855712890625,
      "surface_pressure_hpa": 973.7844848632812,
      "cloud_cover_pct": 95.55270385742188,
      "wind_speed_10m_kmh": 12.77564811706543,
      "rain_probability": 0.8421469330787659,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.009295819327235222,
        "class_1": 0.20261473953723907,
        "class_2": 0.012845687568187714,
        "class_3": 0.21367891132831573,
        "class_4": 0.11506513506174088,
        "class_5": 0.4464803636074066,
        "class_6": 1.9363311366760172e-05
      }
    },
    {
      "lead_hours": 8,
      "target_utc": "2026-05-09T03:00:00+00:00",
      "target_local": "2026-05-09T02:00:00-01:00",
      "temperature_2m_c": 3.953939437866211,
      "relative_humidity_2m_pct": 94.34648895263672,
      "apparent_temperature_c": 0.3805828094482422,
      "precipitation_mm": 0.11994519084692001,
      "pressure_msl_hpa": 1005.9537353515625,
      "surface_pressure_hpa": 973.9803466796875,
      "cloud_cover_pct": 95.32868957519531,
      "wind_speed_10m_kmh": 12.735028266906738,
      "rain_probability": 0.8208134174346924,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.011135051026940346,
        "class_1": 0.21503449976444244,
        "class_2": 0.015187171287834644,
        "class_3": 0.18286095559597015,
        "class_4": 0.1143169179558754,
        "class_5": 0.46144354343414307,
        "class_6": 2.182190291932784e-05
      }
    },
    {
      "lead_hours": 9,
      "target_utc": "2026-05-09T04:00:00+00:00",
      "target_local": "2026-05-09T03:00:00-01:00",
      "temperature_2m_c": 3.826430320739746,
      "relative_humidity_2m_pct": 94.16539001464844,
      "apparent_temperature_c": 0.16225242614746094,
      "precipitation_mm": 0.11211992800235748,
      "pressure_msl_hpa": 1006.1436767578125,
      "surface_pressure_hpa": 974.2029418945312,
      "cloud_cover_pct": 94.6148681640625,
      "wind_speed_10m_kmh": 12.761141777038574,
      "rain_probability": 0.8099679350852966,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.013121353462338448,
        "class_1": 0.23102521896362305,
        "class_2": 0.017765367403626442,
        "class_3": 0.1780213713645935,
        "class_4": 0.12005121260881424,
        "class_5": 0.4399879574775696,
        "class_6": 2.7478810807224363e-05
      }
    },
    {
      "lead_hours": 10,
      "target_utc": "2026-05-09T05:00:00+00:00",
      "target_local": "2026-05-09T04:00:00-01:00",
      "temperature_2m_c": 3.8089590072631836,
      "relative_humidity_2m_pct": 93.53528594970703,
      "apparent_temperature_c": 0.12103080749511719,
      "precipitation_mm": 0.10691206157207489,
      "pressure_msl_hpa": 1006.42529296875,
      "surface_pressure_hpa": 974.4669189453125,
      "cloud_cover_pct": 93.8226318359375,
      "wind_speed_10m_kmh": 12.700864791870117,
      "rain_probability": 0.797008216381073,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.014465805143117905,
        "class_1": 0.22260957956314087,
        "class_2": 0.018675586208701134,
        "class_3": 0.15951745212078094,
        "class_4": 0.10494350641965866,
        "class_5": 0.47975844144821167,
        "class_6": 2.9636566978297196e-05
      }
    },
    {
      "lead_hours": 11,
      "target_utc": "2026-05-09T06:00:00+00:00",
      "target_local": "2026-05-09T05:00:00-01:00",
      "temperature_2m_c": 3.9785900115966797,
      "relative_humidity_2m_pct": 92.22869110107422,
      "apparent_temperature_c": 0.24558448791503906,
      "precipitation_mm": 0.10196752846240997,
      "pressure_msl_hpa": 1006.7659301757812,
      "surface_pressure_hpa": 974.8555297851562,
      "cloud_cover_pct": 92.74380493164062,
      "wind_speed_10m_kmh": 12.71017837524414,
      "rain_probability": 0.7881969213485718,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.01727476716041565,
        "class_1": 0.232261523604393,
        "class_2": 0.019182473421096802,
        "class_3": 0.15716883540153503,
        "class_4": 0.10425136238336563,
        "class_5": 0.46981269121170044,
        "class_6": 4.833983985008672e-05
      }
    },
    {
      "lead_hours": 12,
      "target_utc": "2026-05-09T07:00:00+00:00",
      "target_local": "2026-05-09T06:00:00-01:00",
      "temperature_2m_c": 4.317435264587402,
      "relative_humidity_2m_pct": 90.45832824707031,
      "apparent_temperature_c": 0.6276102066040039,
      "precipitation_mm": 0.09439859539270401,
      "pressure_msl_hpa": 1007.0864868164062,
      "surface_pressure_hpa": 975.1940307617188,
      "cloud_cover_pct": 91.2315673828125,
      "wind_speed_10m_kmh": 12.740204811096191,
      "rain_probability": 0.7812836170196533,
      "weather_class": 5,
      "weather_class_name": "class_5",
      "weather_class_probabilities": {
        "class_0": 0.019589632749557495,
        "class_1": 0.23847728967666626,
        "class_2": 0.020689163357019424,
        "class_3": 0.159035325050354,
        "class_4": 0.08879318088293076,
        "class_5": 0.47335243225097656,
        "class_6": 6.291128374869004e-05
      }
    }
  ],
  "sanity": {
    "sequence_shape": [
      72,
      22
    ],
    "finite_features": true
  }
}
```

### Note
In observed outputs, the model is often within **1°C** of the actual value, which is **0.7** more than Hweh-6M.

Furthermore, you can pass locations that are not present in the model’s location embedding table. We’ve observed that the model can generalize to out-of-distribution (OOD) cities, with an estimated accuracy drop of only about 2–5%. However, this figure is an estimate and does not reflect a true ground-truth measurement.

## Use Cases

Intended for:

1. Backup to API
2. Offline forecasting if you have the data
3. Research
4. Or more simply, for fun

Not intended for:

1. Safety-critical forecasting (aviation, emergency response)
2. Replacing meteorological or API services

## Limitations

1. The model is not perfectly accurate and will produce approximate forecasts rather than exact real-world weather conditions.
2. Prediction accuracy decreases as the forecast horizon increases up to 12 hours.
3. Performance may degrade on unseen or underrepresented geographic regions and climate types.
4. The model does not enforce physical laws of atmospheric dynamics and may produce physically inconsistent outputs.
5. Forecast quality is sensitive to the quality and completeness of input weather data.
6. Rare or extreme weather events are underrepresented in training data and may be poorly predicted.
7. Weather class outputs are simplified and do not capture fine-grained meteorological distinctions.

# Inference

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoConfig, AutoModel
from zoneinfo import ZoneInfo

# ----------------------------
# Change these values here
# ----------------------------
MODEL_ID = r"Harley-ml/Hweh-446k"   # HF repo id or local path
CITY = "New York"
SEQUENCE_META_PATH = "Harley-ml/Hweh-446k/weather_sequences.metadata.json"
CONTEXT_HOURS = 72
FORECAST_HOURS = 12
DEVICE = None  # "cpu", "cuda", "cuda:0", or None for auto

API_BASE_URL = "https://api.open-meteo.com/v1/forecast"
MAX_RETRIES = 6
REQUEST_TIMEOUT_S = 60

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
]

WEATHER_CODE_BUCKETS = 7
TEMP_SCALE = 50.0
HUMIDITY_SCALE = 100.0
WIND_SCALE = 100.0

# ----------------------------
# City metadata (82 locations)
# ----------------------------
CITY_SPECS: dict[str, dict[str, Any]] = {
    "Seattle": {"location_id": "1", "latitude": 47.6062, "longitude": -122.3321, "continent": "North America", "climate_tag": "temperate_oceanic", "elevation": 56},
    "Portland": {"location_id": "2", "latitude": 45.5152, "longitude": -122.6784, "continent": "North America", "climate_tag": "temperate_oceanic", "elevation": 15},
    "San Francisco": {"location_id": "3", "latitude": 37.7749, "longitude": -122.4194, "continent": "North America", "climate_tag": "foggy_mediterranean", "elevation": 16},
    "Los Angeles": {"location_id": "4", "latitude": 34.0522, "longitude": -118.2437, "continent": "North America", "climate_tag": "sunny_mediterranean", "elevation": 71},
    "Denver": {"location_id": "5", "latitude": 39.7392, "longitude": -104.9903, "continent": "North America", "climate_tag": "semi_arid_highland", "elevation": 1609},
    "Chicago": {"location_id": "6", "latitude": 41.8781, "longitude": -87.6298, "continent": "North America", "climate_tag": "humid_continental", "elevation": 181},
    "Dallas": {"location_id": "7", "latitude": 32.7767, "longitude": -96.7970, "continent": "North America", "climate_tag": "hot_subhumid", "elevation": 131},
    "Atlanta": {"location_id": "8", "latitude": 33.7490, "longitude": -84.3880, "continent": "North America", "climate_tag": "humid_subtropical", "elevation": 320},
    "New York": {"location_id": "9", "latitude": 40.7128, "longitude": -74.0060, "continent": "North America", "climate_tag": "humid_subtropical", "elevation": 10},
    "Miami": {"location_id": "10", "latitude": 25.7617, "longitude": -80.1918, "continent": "North America", "climate_tag": "tropical_humid", "elevation": 2},
    "Phoenix": {"location_id": "11", "latitude": 33.4484, "longitude": -112.0740, "continent": "North America", "climate_tag": "hot_arid", "elevation": 331},
    "Salt Lake City": {"location_id": "12", "latitude": 40.7608, "longitude": -111.8910, "continent": "North America", "climate_tag": "semi_arid", "elevation": 1288},
    "Anchorage": {"location_id": "13", "latitude": 61.2181, "longitude": -149.9003, "continent": "North America", "climate_tag": "subarctic_snowy", "elevation": 31},
    "Minneapolis": {"location_id": "14", "latitude": 44.9778, "longitude": -93.2650, "continent": "North America", "climate_tag": "cold_snowy", "elevation": 264},
    "Toronto": {"location_id": "15", "latitude": 43.6532, "longitude": -79.3832, "continent": "North America", "climate_tag": "humid_continental", "elevation": 76},
    "Montreal": {"location_id": "16", "latitude": 45.5017, "longitude": -73.5673, "continent": "North America", "climate_tag": "cold_snowy", "elevation": 233},
    "Vancouver": {"location_id": "17", "latitude": 49.2827, "longitude": -123.1207, "continent": "North America", "climate_tag": "temperate_oceanic", "elevation": 70},
    "Mexico City": {"location_id": "18", "latitude": 19.4326, "longitude": -99.1332, "continent": "North America", "climate_tag": "highland_subtropical", "elevation": 2240},
    "Havana": {"location_id": "19", "latitude": 23.1136, "longitude": -82.3666, "continent": "North America", "climate_tag": "tropical_humid", "elevation": 59},
    "San Juan": {"location_id": "20", "latitude": 18.4655, "longitude": -66.1057, "continent": "North America", "climate_tag": "tropical_humid", "elevation": 8},

    "Lima": {"location_id": "21", "latitude": -12.0464, "longitude": -77.0428, "continent": "South America", "climate_tag": "coastal_arid", "elevation": 154},
    "Santiago": {"location_id": "22", "latitude": -33.4489, "longitude": -70.6693, "continent": "South America", "climate_tag": "mediterranean", "elevation": 520},
    "Buenos Aires": {"location_id": "23", "latitude": -34.6037, "longitude": -58.3816, "continent": "South America", "climate_tag": "humid_subtropical", "elevation": 25},
    "Bogotá": {"location_id": "24", "latitude": 4.7110, "longitude": -74.0721, "continent": "South America", "climate_tag": "highland_cool", "elevation": 2640},
    "Quito": {"location_id": "25", "latitude": -0.1807, "longitude": -78.4678, "continent": "South America", "climate_tag": "highland_equatorial", "elevation": 2850},
    "Caracas": {"location_id": "26", "latitude": 10.4806, "longitude": -66.9036, "continent": "South America", "climate_tag": "tropical_humid", "elevation": 900},
    "Rio de Janeiro": {"location_id": "27", "latitude": -22.9068, "longitude": -43.1729, "continent": "South America", "climate_tag": "tropical_humid", "elevation": 5},
    "São Paulo": {"location_id": "28", "latitude": -23.5505, "longitude": -46.6333, "continent": "South America", "climate_tag": "humid_subtropical", "elevation": 760},
    "La Paz": {"location_id": "29", "latitude": -16.4897, "longitude": -68.1193, "continent": "South America", "climate_tag": "highland_cold", "elevation": 3640},
    "Cusco": {"location_id": "30", "latitude": -13.5319, "longitude": -71.9675, "continent": "South America", "climate_tag": "highland_cool", "elevation": 3399},
    "Montevideo": {"location_id": "31", "latitude": -34.9011, "longitude": -56.1645, "continent": "South America", "climate_tag": "temperate_oceanic", "elevation": 43},
    "Asunción": {"location_id": "32", "latitude": -25.2637, "longitude": -57.5759, "continent": "South America", "climate_tag": "humid_subtropical", "elevation": 43},
    "Manaus": {"location_id": "33", "latitude": -3.1190, "longitude": -60.0217, "continent": "South America", "climate_tag": "tropical_humid", "elevation": 92},
    "Recife": {"location_id": "34", "latitude": -8.0476, "longitude": -34.8770, "continent": "South America", "climate_tag": "tropical_coastal", "elevation": 4},
    "Punta Arenas": {"location_id": "35", "latitude": -53.1638, "longitude": -70.9171, "continent": "South America", "climate_tag": "cold_windy", "elevation": 34},

    "London": {"location_id": "36", "latitude": 51.5074, "longitude": -0.1278, "continent": "Europe", "climate_tag": "temperate_oceanic", "elevation": 11},
    "Paris": {"location_id": "37", "latitude": 48.8566, "longitude": 2.3522, "continent": "Europe", "climate_tag": "temperate_oceanic", "elevation": 35},
    "Madrid": {"location_id": "38", "latitude": 40.4168, "longitude": -3.7038, "continent": "Europe", "climate_tag": "hot_summer_mediterranean", "elevation": 667},
    "Rome": {"location_id": "39", "latitude": 41.9028, "longitude": 12.4964, "continent": "Europe", "climate_tag": "hot_summer_mediterranean", "elevation": 21},
    "Berlin": {"location_id": "40", "latitude": 52.52, "longitude": 13.4050, "continent": "Europe", "climate_tag": "temperate_continental", "elevation": 34},
    "Stockholm": {"location_id": "41", "latitude": 59.3293, "longitude": 18.0686, "continent": "Europe", "climate_tag": "cold_marine", "elevation": 28},
    "Oslo": {"location_id": "42", "latitude": 59.9139, "longitude": 10.7522, "continent": "Europe", "climate_tag": "cold_snowy", "elevation": 23},
    "Helsinki": {"location_id": "43", "latitude": 60.1699, "longitude": 24.9384, "continent": "Europe", "climate_tag": "cold_snowy", "elevation": 25},
    "Reykjavik": {"location_id": "44", "latitude": 64.1466, "longitude": -21.9426, "continent": "Europe", "climate_tag": "cold_windy", "elevation": 12},
    "Kyiv": {"location_id": "45", "latitude": 50.4501, "longitude": 30.5234, "continent": "Europe", "climate_tag": "humid_continental", "elevation": 179},
    "Lisbon": {"location_id": "46", "latitude": 38.7223, "longitude": -9.1393, "continent": "Europe", "climate_tag": "sunny_mediterranean", "elevation": 7},
    "Athens": {"location_id": "47", "latitude": 37.9838, "longitude": 23.7275, "continent": "Europe", "climate_tag": "sunny_mediterranean", "elevation": 70},
    "Zurich": {"location_id": "48", "latitude": 47.3769, "longitude": 8.5417, "continent": "Europe", "climate_tag": "temperate_continental", "elevation": 408},
    "Dublin": {"location_id": "49", "latitude": 53.3498, "longitude": -6.2603, "continent": "Europe", "climate_tag": "temperate_oceanic", "elevation": 20},
    "Vienna": {"location_id": "50", "latitude": 48.2082, "longitude": 16.3738, "continent": "Europe", "climate_tag": "temperate_continental", "elevation": 171},

    "Dubai": {"location_id": "51", "latitude": 25.2048, "longitude": 55.2708, "continent": "Asia", "climate_tag": "hot_arid", "elevation": 16},
    "Riyadh": {"location_id": "52", "latitude": 24.7136, "longitude": 46.6753, "continent": "Asia", "climate_tag": "hot_arid", "elevation": 612},
    "Delhi": {"location_id": "53", "latitude": 28.7041, "longitude": 77.1025, "continent": "Asia", "climate_tag": "hot_semi_arid", "elevation": 216},
    "Mumbai": {"location_id": "54", "latitude": 19.0760, "longitude": 72.8777, "continent": "Asia", "climate_tag": "tropical_humid", "elevation": 14},
    "Bangkok": {"location_id": "55", "latitude": 13.7563, "longitude": 100.5018, "continent": "Asia", "climate_tag": "tropical_monsoon", "elevation": 2},
    "Singapore": {"location_id": "56", "latitude": 1.3521, "longitude": 103.8198, "continent": "Asia", "climate_tag": "tropical_humid", "elevation": 15},
    "Tokyo": {"location_id": "57", "latitude": 35.6762, "longitude": 139.6503, "continent": "Asia", "climate_tag": "humid_subtropical", "elevation": 40},
    "Seoul": {"location_id": "58", "latitude": 37.5665, "longitude": 126.9780, "continent": "Asia", "climate_tag": "humid_continental", "elevation": 38},
    "Ulaanbaatar": {"location_id": "59", "latitude": 47.8864, "longitude": 106.9057, "continent": "Asia", "climate_tag": "cold_steppe", "elevation": 1350},
    "Kathmandu": {"location_id": "60", "latitude": 27.7172, "longitude": 85.3240, "continent": "Asia", "climate_tag": "highland_subtropical", "elevation": 1400},
    "Chiang Mai": {"location_id": "61", "latitude": 18.7883, "longitude": 98.9853, "continent": "Asia", "climate_tag": "tropical_seasonal", "elevation": 300},
    "Lhasa": {"location_id": "62", "latitude": 29.6520, "longitude": 91.1721, "continent": "Asia", "climate_tag": "high_altitude_cold", "elevation": 3656},
    "Jakarta": {"location_id": "63", "latitude": -6.2088, "longitude": 106.8456, "continent": "Asia", "climate_tag": "tropical_humid", "elevation": 8},
    "Manila": {"location_id": "64", "latitude": 14.5995, "longitude": 120.9842, "continent": "Asia", "climate_tag": "tropical_humid", "elevation": 16},
    "Karachi": {"location_id": "65", "latitude": 24.8607, "longitude": 67.0011, "continent": "Asia", "climate_tag": "hot_arid", "elevation": 10},

    "Cairo": {"location_id": "66", "latitude": 30.0444, "longitude": 31.2357, "continent": "Africa", "climate_tag": "hot_arid", "elevation": 23},
    "Alexandria": {"location_id": "67", "latitude": 31.2001, "longitude": 29.9187, "continent": "Africa", "climate_tag": "coastal_mediterranean", "elevation": 5},
    "Casablanca": {"location_id": "68", "latitude": 33.5731, "longitude": -7.5898, "continent": "Africa", "climate_tag": "coastal_mediterranean", "elevation": 56},
    "Marrakech": {"location_id": "69", "latitude": 31.6295, "longitude": -7.9811, "continent": "Africa", "climate_tag": "hot_semi_arid", "elevation": 466},
    "Lagos": {"location_id": "70", "latitude": 6.5244, "longitude": 3.3792, "continent": "Africa", "climate_tag": "tropical_humid", "elevation": 41},
    "Nairobi": {"location_id": "71", "latitude": -1.2921, "longitude": 36.8219, "continent": "Africa", "climate_tag": "temperate_highland", "elevation": 1795},
    "Addis Ababa": {"location_id": "72", "latitude": 8.9806, "longitude": 38.7578, "continent": "Africa", "climate_tag": "temperate_highland", "elevation": 2355},
    "Cape Town": {"location_id": "73", "latitude": -33.9249, "longitude": 18.4241, "continent": "Africa", "climate_tag": "mediterranean", "elevation": 25},
    "Johannesburg": {"location_id": "74", "latitude": -26.2041, "longitude": 28.0473, "continent": "Africa", "climate_tag": "subtropical_highland", "elevation": 1753},
    "Windhoek": {"location_id": "75", "latitude": -22.5609, "longitude": 17.0658, "continent": "Africa", "climate_tag": "semi_arid", "elevation": 1650},
    "Accra": {"location_id": "76", "latitude": 5.6037, "longitude": -0.1870, "continent": "Africa", "climate_tag": "tropical_humid", "elevation": 61},
    "Kigali": {"location_id": "77", "latitude": -1.9441, "longitude": 30.0619, "continent": "Africa", "climate_tag": "highland_tropical", "elevation": 1567},
    "Tunis": {"location_id": "78", "latitude": 36.8065, "longitude": 10.1815, "continent": "Africa", "climate_tag": "mediterranean", "elevation": 4},
    "Dakar": {"location_id": "79", "latitude": -14.7167, "longitude": -17.4677, "continent": "Africa", "climate_tag": "hot_coastal", "elevation": 25},
    "Mombasa": {"location_id": "80", "latitude": -4.0435, "longitude": 39.6682, "continent": "Africa", "climate_tag": "tropical_coastal", "elevation": 17},

    "Sydney": {"location_id": "81", "latitude": -33.8688, "longitude": 151.2093, "continent": "Oceania", "climate_tag": "humid_subtropical", "elevation": 58},
    "Melbourne": {"location_id": "82", "latitude": -37.8136, "longitude": 144.9631, "continent": "Oceania", "climate_tag": "temperate_oceanic", "elevation": 31},
}

CITY_TIMEZONES: dict[str, str] = {
    "Seattle": "America/Los_Angeles",
    "Portland": "America/Los_Angeles",
    "San Francisco": "America/Los_Angeles",
    "Los Angeles": "America/Los_Angeles",
    "Denver": "America/Denver",
    "Chicago": "America/Chicago",
    "Dallas": "America/Chicago",
    "Atlanta": "America/New_York",
    "New York": "America/New_York",
    "Miami": "America/New_York",
    "Phoenix": "America/Phoenix",
    "Salt Lake City": "America/Denver",
    "Anchorage": "America/Anchorage",
    "Minneapolis": "America/Chicago",
    "Toronto": "America/Toronto",
    "Montreal": "America/Toronto",
    "Vancouver": "America/Vancouver",
    "Mexico City": "America/Mexico_City",
    "Havana": "America/Havana",
    "San Juan": "America/Puerto_Rico",
    "Lima": "America/Lima",
    "Santiago": "America/Santiago",
    "Buenos Aires": "America/Argentina/Buenos_Aires",
    "Bogotá": "America/Bogota",
    "Quito": "America/Guayaquil",
    "Caracas": "America/Caracas",
    "Rio de Janeiro": "America/Sao_Paulo",
    "São Paulo": "America/Sao_Paulo",
    "La Paz": "America/La_Paz",
    "Cusco": "America/Lima",
    "Montevideo": "America/Montevideo",
    "Asunción": "America/Asuncion",
    "Manaus": "America/Manaus",
    "Recife": "America/Recife",
    "Punta Arenas": "America/Punta_Arenas",
    "London": "Europe/London",
    "Paris": "Europe/Paris",
    "Madrid": "Europe/Madrid",
    "Rome": "Europe/Rome",
    "Berlin": "Europe/Berlin",
    "Stockholm": "Europe/Stockholm",
    "Oslo": "Europe/Oslo",
    "Helsinki": "Europe/Helsinki",
    "Reykjavik": "Atlantic/Reykjavik",
    "Kyiv": "Europe/Kyiv",
    "Lisbon": "Europe/Lisbon",
    "Athens": "Europe/Athens",
    "Zurich": "Europe/Zurich",
    "Dublin": "Europe/Dublin",
    "Vienna": "Europe/Vienna",
    "Dubai": "Asia/Dubai",
    "Riyadh": "Asia/Riyadh",
    "Delhi": "Asia/Kolkata",
    "Mumbai": "Asia/Kolkata",
    "Bangkok": "Asia/Bangkok",
    "Singapore": "Asia/Singapore",
    "Tokyo": "Asia/Tokyo",
    "Seoul": "Asia/Seoul",
    "Ulaanbaatar": "Asia/Ulaanbaatar",
    "Kathmandu": "Asia/Kathmandu",
    "Chiang Mai": "Asia/Bangkok",
    "Lhasa": "Asia/Shanghai",
    "Jakarta": "Asia/Jakarta",
    "Manila": "Asia/Manila",
    "Karachi": "Asia/Karachi",
    "Cairo": "Africa/Cairo",
    "Alexandria": "Africa/Cairo",
    "Casablanca": "Africa/Casablanca",
    "Marrakech": "Africa/Casablanca",
    "Lagos": "Africa/Lagos",
    "Nairobi": "Africa/Nairobi",
    "Addis Ababa": "Africa/Addis_Ababa",
    "Cape Town": "Africa/Johannesburg",
    "Johannesburg": "Africa/Johannesburg",
    "Windhoek": "Africa/Windhoek",
    "Accra": "Africa/Accra",
    "Kigali": "Africa/Kigali",
    "Tunis": "Africa/Tunis",
    "Dakar": "Africa/Dakar",
    "Mombasa": "Africa/Nairobi",
    "Sydney": "Australia/Sydney",
    "Melbourne": "Australia/Melbourne",
}

# ----------------------------
# Helpers
# ----------------------------
def weather_code_to_bucket(code) -> int:
    if code is None:
        return 1
    try:
        if pd.isna(code):
            return 1
    except Exception:
        pass

    code = int(code)
    if code == 0:
        return 0
    if code in (1, 2, 3):
        return 1
    if code in (45, 48):
        return 2
    if code in (51, 53, 55, 56, 57):
        return 3
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return 4
    if code in (71, 73, 75, 77, 85, 86):
        return 5
    if code in (95, 96, 99):
        return 6
    return 1


def cyc(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * (x / period)
    return np.sin(angle), np.cos(angle)


def clamp_array(x: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    return np.clip(x, lo, hi)


def request_with_backoff(session: requests.Session, url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else min(60.0, 2**attempt)
                print(f"Rate limited. Sleeping {sleep_s:.1f}s and retrying.", flush=True)
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            sleep_s = min(60.0, 2**attempt)
            print(f"Request failed: {e}. Sleeping {sleep_s:.1f}s and retrying.", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {params}") from last_exc


def load_sequence_meta(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"location_to_id": {}}
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.setdefault("location_to_id", {})
    return meta


def load_model():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, config=config, trust_remote_code=True)
    model.eval()
    return model, config


def fetch_recent_history(city: str, context_hours: int) -> pd.DataFrame:
    if city not in CITY_SPECS:
        raise ValueError(f"Unknown city: {city}")

    spec = CITY_SPECS[city]
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    params = {
        "latitude": spec["latitude"],
        "longitude": spec["longitude"],
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "past_hours": int(context_hours) + 2,
        "forecast_hours": 0,
    }

    data = request_with_backoff(session, API_BASE_URL, params=params)
    hourly = data.get("hourly", {})
    if "time" not in hourly:
        raise ValueError(f"No hourly data returned for {city}: {data}")

    df = pd.DataFrame(hourly)
    if df.empty:
        raise ValueError(f"Empty hourly response for {city}.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    needed = HOURLY_VARS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing hourly columns in API response: {missing}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["weather_code"] = df["weather_code"].fillna(1)
    df["precipitation"] = df["precipitation"].fillna(0.0)

    for c in [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "visibility",
        "wind_speed_10m",
        "wind_direction_10m",
    ]:
        df[c] = df[c].interpolate(limit_direction="both").ffill().bfill()

    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["time"] <= now_utc].copy()

    if len(df) < context_hours:
        raise ValueError(f"Not enough observed rows: got {len(df)}, need {context_hours}")

    return df.tail(context_hours).reset_index(drop=True)


def build_single_sequence(df: pd.DataFrame) -> np.ndarray:
    hour = df["time"].dt.hour.to_numpy()
    doy = df["time"].dt.dayofyear.to_numpy()

    hour_sin, hour_cos = cyc(hour.astype(float), 24.0)
    doy_sin, doy_cos = cyc(doy.astype(float), 365.25)

    temp = np.nan_to_num(df["temperature_2m"].astype(float).to_numpy(), nan=0.0)
    humidity = np.nan_to_num(df["relative_humidity_2m"].astype(float).to_numpy(), nan=0.0)
    apparent = np.nan_to_num(df["apparent_temperature"].astype(float).to_numpy(), nan=0.0)
    precip = np.nan_to_num(df["precipitation"].astype(float).to_numpy(), nan=0.0)
    pressure = np.nan_to_num(df["pressure_msl"].astype(float).to_numpy(), nan=0.0)
    surface_pressure = np.nan_to_num(df["surface_pressure"].astype(float).to_numpy(), nan=0.0)
    cloud_cover = np.nan_to_num(df["cloud_cover"].astype(float).to_numpy(), nan=0.0)
    visibility = np.nan_to_num(df["visibility"].astype(float).to_numpy(), nan=0.0)
    wind = np.nan_to_num(df["wind_speed_10m"].astype(float).to_numpy(), nan=0.0)
    wind_dir = np.nan_to_num(df["wind_direction_10m"].astype(float).to_numpy(), nan=0.0)

    humidity = clamp_array(humidity, 0.0, 100.0)
    cloud_cover = clamp_array(cloud_cover, 0.0, 100.0)
    precip = clamp_array(precip, 0.0, None)
    wind = clamp_array(wind, 0.0, None)
    visibility = clamp_array(visibility, 0.0, None)

    wind_dir_sin, wind_dir_cos = cyc(wind_dir, 360.0)
    weather_bucket = df["weather_code"].fillna(1).apply(weather_code_to_bucket).to_numpy(dtype=np.int64)

    rows = []
    for i in range(len(df)):
        wc_oh = np.zeros(WEATHER_CODE_BUCKETS, dtype=np.float32)
        wc_oh[weather_bucket[i]] = 1.0

        row = np.concatenate(
            [
                np.array(
                    [
                        temp[i] / TEMP_SCALE,
                        humidity[i] / HUMIDITY_SCALE,
                        apparent[i] / TEMP_SCALE,
                        np.log1p(max(precip[i], 0.0)) / 3.0,
                        pressure[i] / 1100.0,
                        surface_pressure[i] / 1100.0,
                        cloud_cover[i] / 100.0,
                        visibility[i] / 50000.0,
                        wind[i] / WIND_SCALE,
                        wind_dir_sin[i],
                        wind_dir_cos[i],
                        hour_sin[i],
                        hour_cos[i],
                        doy_sin[i],
                        doy_cos[i],
                    ],
                    dtype=np.float32,
                ),
                wc_oh,
            ]
        )
        rows.append(row)

    seq = np.asarray(rows, dtype=np.float32)

    if not np.isfinite(seq).all():
        bad = np.argwhere(~np.isfinite(seq))
        raise ValueError(f"Non-finite values remain in sequence at positions like: {bad[:10].tolist()}")

    return seq


def to_iso(ts: pd.Timestamp, tz_name: str | None = None) -> str:
    if tz_name:
        try:
            return ts.tz_convert(ZoneInfo(tz_name)).isoformat()
        except Exception:
            pass
    return ts.isoformat()


def get_logits(out):
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    if hasattr(out, "logits"):
        return out.logits
    return out


def resolve_location_index(seq_meta: dict[str, Any], city_location_id: str) -> int:
    location_to_id = seq_meta.get("location_to_id", {})

    if city_location_id in location_to_id:
        return int(location_to_id[city_location_id])

    try:
        as_int = int(city_location_id)
        if as_int in location_to_id:
            return int(location_to_id[as_int])
        if str(as_int) in location_to_id:
            return int(location_to_id[str(as_int)])
    except Exception:
        pass

    for unk_key in ("UNK", "<UNK>", "unknown", "UNKNOWN"):
        if unk_key in location_to_id:
            return int(location_to_id[unk_key])

    return 0


def predict():
    seq_meta = load_sequence_meta(SEQUENCE_META_PATH)
    model, config = load_model()

    if CITY not in CITY_SPECS:
        raise ValueError(f"Unknown city: {CITY}")

    if CONTEXT_HOURS <= 0:
        raise ValueError("CONTEXT_HOURS must be > 0")

    if hasattr(config, "seq_len") and int(config.seq_len) != CONTEXT_HOURS:
        raise ValueError(f"Set CONTEXT_HOURS to {int(config.seq_len)} for this model.")

    city_spec = CITY_SPECS[CITY]
    city_tz = CITY_TIMEZONES.get(CITY, "UTC")
    model_location_id = resolve_location_index(seq_meta, str(city_spec["location_id"]))

    df = fetch_recent_history(CITY, CONTEXT_HOURS)
    seq = build_single_sequence(df)

    X = torch.from_numpy(seq).unsqueeze(0)
    loc = torch.tensor([model_location_id], dtype=torch.long)

    target_device = torch.device(
        DEVICE if DEVICE else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(target_device)
    X = X.to(target_device)
    loc = loc.to(target_device)

    weather_class_names = getattr(config, "weather_class_names", None)
    if not weather_class_names:
        weather_class_names = [f"class_{i}" for i in range(int(getattr(config, "num_weather_classes", 7)))]

    with torch.no_grad():
        out = model(X=X, location_id=loc)
        logits = get_logits(out)

        (
            temp_pred,
            humidity_pred,
            apparent_pred,
            precip_pred,
            sea_level_pressure_pred,
            surface_pressure_pred,
            cloud_cover_pred,
            wind_pred,
            wind_dir_sin_pred,
            wind_dir_cos_pred,
            rain_logit,
            weather_logits,
        ) = logits

        temp_pred = temp_pred.squeeze(0).detach().cpu().numpy()
        humidity_pred = humidity_pred.squeeze(0).detach().cpu().numpy()
        apparent_pred = apparent_pred.squeeze(0).detach().cpu().numpy()
        precip_pred = precip_pred.squeeze(0).detach().cpu().numpy()
        sea_level_pressure_pred = sea_level_pressure_pred.squeeze(0).detach().cpu().numpy()
        surface_pressure_pred = surface_pressure_pred.squeeze(0).detach().cpu().numpy()
        cloud_cover_pred = cloud_cover_pred.squeeze(0).detach().cpu().numpy()
        wind_pred = wind_pred.squeeze(0).detach().cpu().numpy()
        rain_prob = torch.sigmoid(rain_logit).squeeze(0).detach().cpu().numpy()
        weather_probs = torch.softmax(weather_logits, dim=-1).squeeze(0).detach().cpu().numpy()
        weather_idx = np.argmax(weather_probs, axis=-1).astype(np.int64)

        humidity_pred = np.clip(humidity_pred, 0.0, 100.0)
        cloud_cover_pred = np.clip(cloud_cover_pred, 0.0, 100.0)
        precip_pred = np.clip(precip_pred, 0.0, None)
        wind_pred = np.clip(wind_pred, 0.0, None)
        rain_prob = np.clip(rain_prob, 0.0, 1.0)

    context_start = df["time"].iloc[0]
    context_end = df["time"].iloc[-1]
    requested_at_utc = pd.Timestamp.now(tz="UTC")

    horizon = min(
        int(FORECAST_HOURS),
        int(temp_pred.shape[0]),
        int(humidity_pred.shape[0]),
        int(weather_idx.shape[0]),
    )

    forecast = []
    for lead in range(1, horizon + 1):
        target_time = context_end + pd.Timedelta(hours=lead)
        idx = lead - 1
        w_idx = int(weather_idx[idx])

        forecast.append(
            {
                "lead_hours": lead,
                "target_utc": target_time.isoformat(),
                "target_local": to_iso(target_time, city_tz),
                "temperature_2m_c": float(temp_pred[idx]),
                "relative_humidity_2m_pct": float(humidity_pred[idx]),
                "apparent_temperature_c": float(apparent_pred[idx]),
                "precipitation_mm": float(precip_pred[idx]),
                "pressure_msl_hpa": float(sea_level_pressure_pred[idx]),
                "surface_pressure_hpa": float(surface_pressure_pred[idx]),
                "cloud_cover_pct": float(cloud_cover_pred[idx]),
                "wind_speed_10m_kmh": float(wind_pred[idx]),
                "rain_probability": float(rain_prob[idx]),
                "weather_class": w_idx,
                "weather_class_name": weather_class_names[w_idx] if w_idx < len(weather_class_names) else f"class_{w_idx}",
                "weather_class_probabilities": {
                    name: float(prob) for name, prob in zip(weather_class_names, weather_probs[idx])
                },
            }
        )

    result = {
        "city": CITY,
        "location_id": str(city_spec["location_id"]),
        "model_location_id": int(model_location_id),
        "data_source": "open-meteo forecast api (past-hours context only)",
        "requested_at_utc": requested_at_utc.isoformat(),
        "context": {
            "hours": int(len(df)),
            "start_utc": context_start.isoformat(),
            "end_utc": context_end.isoformat(),
            "start_local": to_iso(context_start, city_tz),
            "end_local": to_iso(context_end, city_tz),
        },
        "model": {
            "model_id": MODEL_ID,
            "encoder_type": getattr(config, "encoder_type", None),
            "seq_len": int(getattr(config, "seq_len", CONTEXT_HOURS)),
            "input_dim": int(getattr(config, "input_dim", seq.shape[1])),
            "num_weather_classes": int(getattr(config, "num_weather_classes", len(weather_class_names))),
        },
        "forecast": forecast,
        "sanity": {
            "sequence_shape": list(seq.shape),
            "finite_features": bool(np.isfinite(seq).all()),
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    predict()
```

### Related Models

1. [Hweh-6M](https://huggingface.co/Harley-ml/Hweh-6M)

## Citation

```bibtex
@misc{distilhweh-446k,
  title = {DistilHweh-446k: Knowledge Distillation in Short-Term Multivariate Weather Forecasting},
  author    = {Paul Courneya; Harley-ml},
  year      = {2026},
  url       = {https://huggingface.co/Harley-ml/DistilHweh-446k}
}
```