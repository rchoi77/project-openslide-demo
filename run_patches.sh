#!/bin/bash

# Usage: ./run_patches.sh WSI/slide_file.tiff

if [ "$#" -le 0 ]; then
    echo "Please provide a slide image path."
    exit 1
fi

X_START=10
X_END=90
Y_START=10
Y_END=90
STEP_SIZE=5
SIZE="1024 1024"

uv run python src/demo-scripts/generate_patches.py \
    -x $X_START $X_END \
    -y $Y_START $Y_END \
    --step-size $STEP_SIZE \
    --size $SIZE \
    "$1"