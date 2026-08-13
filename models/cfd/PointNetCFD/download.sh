#!/bin/bash
set -e

cd "$(dirname "$0")"

# Download the weight directory from ModelScope into ./weight.
modelscope download --model OneScience/PointNetCFD weight --local_dir ./
