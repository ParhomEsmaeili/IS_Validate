#!/bin/bash

# Array of jobs: each job runs offline_sampling_split.py on a dataset folder with desired strategy and split
jobs=(
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset001_BrainTumour' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset004_Hippocampus' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset005_Prostate' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset006_Lung' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset008_HepaticVessel' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset019CorruptionMethodadd_gaussCorruptionParam200_Liver' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset020CorruptionMethodadd_gaussCorruptionParam400_Liver' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset021CorruptionMethodadd_gaussCorruptionParam600_Liver' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset022CorruptionMethodadd_gaussCorruptionParam800_Liver' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset023CorruptionMethodadd_gaussCorruptionParam200_Pancreas' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset024CorruptionMethodadd_gaussCorruptionParam400_Pancreas' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset025CorruptionMethodadd_gaussCorruptionParam600_Pancreas' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset026CorruptionMethodadd_gaussCorruptionParam800_Pancreas' --strategy=all --split=train"
)

# Run jobs sequentially
for job in "${jobs[@]}"; do
  echo "Running: $job"
  eval "$job"
done