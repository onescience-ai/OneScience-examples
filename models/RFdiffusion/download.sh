#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 10 个大文件
modelscope download --model OneScience/RFdiffusion img/rfdiffusion_logo1.png --local_dir ./
modelscope download --model OneScience/RFdiffusion models/InpaintSeq_ckpt.pt --local_dir ./
modelscope download --model OneScience/RFdiffusion models/Base_ckpt.pt --local_dir ./
modelscope download --model OneScience/RFdiffusion models/Complex_Fold_base_ckpt.pt --local_dir ./
modelscope download --model OneScience/RFdiffusion models/Complex_beta_ckpt.pt --local_dir ./
modelscope download --model OneScience/RFdiffusion models/ActiveSite_ckpt.pt --local_dir ./
modelscope download --model OneScience/RFdiffusion models/Complex_base_ckpt.pt --local_dir ./
modelscope download --model OneScience/RFdiffusion models/Base_epoch8_ckpt.pt --local_dir ./
modelscope download --model OneScience/RFdiffusion models/InpaintSeq_Fold_ckpt.pt --local_dir ./
modelscope download --model OneScience/RFdiffusion models/RF_structure_prediction_weights.pt --local_dir ./
