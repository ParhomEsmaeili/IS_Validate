#!/bin/bash

# Array of jobs: each job runs offline_sampling_split.py on a dataset folder with desired strategy and split
# jobs=(
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset001_BrainTumour' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset002_Heart' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset004_Hippocampus' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset005_Prostate' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset006_Lung' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset008_HepaticVessel' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset009_Spleen' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset010_Colon' --strategy=all --split=train"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset001_BrainTumour' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset002_Heart' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset004_Hippocampus' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset005_Prostate' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset006_Lung' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset008_HepaticVessel' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset009_Spleen' --strategy=all --split=test"
#   "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset010_Colon' --strategy=all --split=test"
#   # "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset010_Colon' --strategy=kfold --split=test"
  
# )

jobs=(
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset011_BrainTumour' --strategy=all --split=train"
  "python3 offline_sampling_split.py --dataset_dir='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset011_BrainTumour' --strategy=all --split=test" 
)
# Run jobs sequentially
for job in "${jobs[@]}"; do
  echo "Running: $job"
  eval "$job"
done