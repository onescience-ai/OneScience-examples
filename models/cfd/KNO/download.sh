#!/bin/bash
# Download files larger than 1MB, excluding .sh .py .md .yaml .yml
# Total large files: 1
modelscope download --model OneScience/KNO weight/kno_navier_stokes.pt --local_dir ./
