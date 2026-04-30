#!/bin/bash

for dataset_dir in /home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/*/; do
    dataset_name=$(basename "$dataset_dir")
    echo "Checking spacing for $dataset_name..."
    python3 check_samples_spacing.py --dataset "$dataset_name"
done