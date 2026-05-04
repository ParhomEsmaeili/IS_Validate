#!/bin/bash

# Array of jobs: each job runs inter_rater_threshold.py on a dataset folder with desired strategy and split

jobs=(
  # "python3 inter_rater_threshold.py --dataset_root '/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/' --dataset_name='Dataset011_Kits23' --class-labels='whole_kidney' --exclude-annotators='annotator_4'"
  
  "python3 inter_rater_threshold.py --dataset_root '/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/' --dataset_name='Dataset015_Kits23' --class-labels='tumor' --exclude-annotators='annotator_4'"
  
  # "python3 inter_rater_threshold.py --dataset_root '/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/' --dataset_name='Dataset019_Kits23' --class-labels='mass' --exclude-annotators='annotator_4'"
)

# Run jobs sequentially
for job in "${jobs[@]}"; do
  echo "Running: $job"
  eval "$job"
done