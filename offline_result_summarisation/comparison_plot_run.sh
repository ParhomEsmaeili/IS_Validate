#!/bin/bash

# Set your arguments here
DATASET_NAME="Dataset007_Pancreas"
DATETIMES=("20250615_102156" "20250614_025617" "20250619_000045" "20250621_205840")
APP_NAMES=("SAMMed2D" "SAM2" "SAMMed3D" "SegVol")

# Build the argument string for datetimes and app names
DATETIMES_ARGS=()
for dt in "${DATETIMES[@]}"; do
    DATETIMES_ARGS+=("$dt")
done

APP_NAMES_ARGS=()
for app in "${APP_NAMES[@]}"; do
    APP_NAMES_ARGS+=("$app")
done

python3 comparison_plot.py \
    --dataset_name "$DATASET_NAME" \
    --datetimes "${DATETIMES_ARGS[@]}" \
    --app_names "${APP_NAMES_ARGS[@]}"