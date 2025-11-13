#!/bin/bash

# Set your arguments here
ROOT_INPUT_PATH="/home/parhomesmaeili/Helmholtz Group/MIDL2025_nnunet/nnUNet_Metrics"
DATASET_NAME=(
    "Dataset001_BrainTumour" \
    "Dataset003_Liver" \
    "Dataset004_Hippocampus" \
    "Dataset005_Prostate" \
    "Dataset006_Lung" \
    "Dataset007_Pancreas" \
    "Dataset008_HepaticVessel" \
    "Dataset010_Colon")
ROOT_OUTPUT_PATH="/home/parhomesmaeili/Helmholtz Group/MIDL2025_nnunet/nnUNet_Metrics_Visualised"
REFERENCE_METRIC="Dice"
REFERENCE_FILE="cross_class_scores.csv"
REFERENCE_COLUMN="Automatic Init"

for DATASET in "${DATASET_NAME[@]}"; do
    python3 nnunet_distribution_visualisation.py \
        --root_input_path "$ROOT_INPUT_PATH" \
        --root_output_path "$ROOT_OUTPUT_PATH" \
        --dataset_name "$DATASET" \
        --reference_metric "$REFERENCE_METRIC" \
        --reference_file "$REFERENCE_FILE" \
        --reference_column "$REFERENCE_COLUMN" \
        ;
done
