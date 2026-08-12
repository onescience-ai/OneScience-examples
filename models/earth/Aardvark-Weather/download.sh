#!/bin/bash
# Download files larger than 1MB and data files, excluding .sh .py .md .yaml .yml
# Total download files: 22
modelscope download --model OneScience/Aardvark-Weather official-src/data/grid_lon_lat/era5_x_1.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/grid_lon_lat/era5_y_1.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/norm_factors/mean_4u_1.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/norm_factors/mean_diff_4u_1.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/norm_factors/mean_hadisd_tas.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/norm_factors/mean_hadisd_ws.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/norm_factors/std_4u_1.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/norm_factors/std_diff_4u_1.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/norm_factors/std_hadisd_tas.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/norm_factors/std_hadisd_ws.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/data/sample_data_final.pkl --local_dir ./
modelscope download --model OneScience/Aardvark-Weather official-src/training/test/config.pkl --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/sample_data/sample_data_final.pkl --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/decoder/tas/config.pkl --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/decoder/tas/lt_1/epoch_18 --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/decoder/tas/lt_1/losses_0.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/encoder/config.pkl --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/encoder/epoch_96 --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/encoder/losses_0.npy --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/processor/config.pkl --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/processor/forecast_1/epoch_0 --local_dir ./
modelscope download --model OneScience/Aardvark-Weather weights/trained_model/processor/forecast_1/losses_0.npy --local_dir ./
