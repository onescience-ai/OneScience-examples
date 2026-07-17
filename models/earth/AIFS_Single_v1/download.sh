#!/bin/bash
# Download files larger than 1MB, excluding .sh .py .md .yaml .yml
# Total large files: 1
modelscope download --model OneScience/AIFS_Single_v1 model/grid-n320.npz --local_dir ./
