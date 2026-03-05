#!/bin/bash

# Array of jobs: each job runs convert_datasplit_to_nnunet.py on a dataset folder with desired strategy and split


jobs=(
  "python3 convert_datasplit_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset001_BrainTumour/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_preprocessed' --split_name='train' --split_type='kfold_5_train'"
  "python3 convert_datasplit_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_preprocessed' --split_name='train' --split_type='kfold_5_train'"
  "python3 convert_datasplit_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset004_Hippocampus/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_preprocessed' --split_name='train' --split_type='kfold_5_train'"
  "python3 convert_datasplit_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset005_Prostate/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_preprocessed' --split_name='train' --split_type='kfold_5_train'"
  "python3 convert_datasplit_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset006_Lung/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_preprocessed' --split_name='train' --split_type='kfold_5_train'"
  "python3 convert_datasplit_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_preprocessed' --split_name='train' --split_type='kfold_5_train'"
  "python3 convert_datasplit_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset008_HepaticVessel/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_preprocessed' --split_name='train' --split_type='kfold_5_train'"
  "python3 convert_datasplit_to_nnunet.py --reference_dataset_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset010_Colon/' --target_dataset_basedir_path='/home/parhomesmaeili/Helmholtz Group/NeurIPS2026_nnunet/nnUNet_preprocessed' --split_name='train' --split_type='kfold_5_train'" 
)

# Run jobs sequentially
for job in "${jobs[@]}"; do
  echo "Running: $job"
  eval "$job"
done