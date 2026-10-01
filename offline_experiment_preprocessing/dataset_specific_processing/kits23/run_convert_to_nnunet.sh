#!/bin/bash

# Array of jobs: each job runs offline_sampling_split.py on a dataset folder with desired strategy and split

jobs=(
  "python3 convert_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset011_Kits23/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_raw' --output_name='Dataset011_Kits23' --task_config_basepath='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs' --split_name='train' --reference_exp_config_ids 36 37 38 39 40"

  "python3 convert_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset015_Kits23/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_raw' --output_name='Dataset015_Kits23' --task_config_basepath='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs' --split_name='train' --reference_exp_config_ids 36 37 38 39 40"

  "python3 convert_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset019_Kits23/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_raw' --output_name='Dataset019_Kits23' --task_config_basepath='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs' --split_name='train' --reference_exp_config_ids 36 37 38 39 40"
  
)

# Run jobs sequentially
for job in "${jobs[@]}"; do
  echo "Running: $job"
  eval "$job"
done