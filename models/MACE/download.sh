#!/bin/bash
# 下载所有 >1MB 且非 .sh .py .md .yaml .yml 的文件
# 共 19 个大文件
modelscope download --model OneScience/MACE data/ani1x/ani1x_test.xyz --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ani1x_cc_dft.xyz --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ani1x_train.xyz --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_test/Default__6.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_test/Default__1.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_test/Default__0.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_test/Default__3.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_test/Default__2.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_test/Default__4.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_test/Default__5.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_test/Default__7.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_train/train_0.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_train/train_5.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_train/train_4.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_train/train_6.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_train/train_7.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_train/train_2.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_train/train_1.h5 --local_dir ./
modelscope download --model OneScience/MACE data/ani1x/ANI1x_cc_DFT_rc5_train/train_3.h5 --local_dir ./
