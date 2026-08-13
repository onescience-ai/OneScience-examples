#!/bin/bash
set -e

cd "$(dirname "$0")"

# Download the weight directory from ModelScope into ./weight.
modelscope download --model OneScience/MP_PDE weight --local_dir ./
