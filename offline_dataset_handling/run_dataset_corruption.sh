#!/bin/bash

jobs=(
  # "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' -dataset_id=11 -corruption_method='multip_gauss' -corruption_param='0.5' -np=8"
  # "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' -dataset_id=12 -corruption_method='multip_gauss' -corruption_param='1' -np=8"
  # "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' -dataset_id=13 -corruption_method='multip_gauss' -corruption_param='1.5' -np=8"
  # "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' -dataset_id=14 -corruption_method='multip_gauss' -corruption_param='2' -np=8"
  # "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' -dataset_id=15 -corruption_method='multip_gauss' -corruption_param='0.5' -np=8"
  # "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' -dataset_id=16 -corruption_method='multip_gauss' -corruption_param='1' -np=8"
  # "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' -dataset_id=17 -corruption_method='multip_gauss' -corruption_param='1.5' -np=8"
  # "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' -dataset_id=18 -corruption_method='multip_gauss' -corruption_param='2' -np=8"
  "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' -dataset_id=19 -corruption_method='add_gauss' -corruption_param='200' -np=8"
  "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' -dataset_id=20 -corruption_method='add_gauss' -corruption_param='400' -np=8"
  "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' -dataset_id=21 -corruption_method='add_gauss' -corruption_param='600' -np=8"
  "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset003_Liver' -dataset_id=22 -corruption_method='add_gauss' -corruption_param='800' -np=8"
  "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' -dataset_id=23 -corruption_method='add_gauss' -corruption_param='200' -np=8"
  "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' -dataset_id=24 -corruption_method='add_gauss' -corruption_param='400' -np=8"
  "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' -dataset_id=25 -corruption_method='add_gauss' -corruption_param='600' -np=8"
  "python3 offline_image_manipulation.py -d_path='/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/Dataset007_Pancreas' -dataset_id=26 -corruption_method='add_gauss' -corruption_param='800' -np=8"
)

# Run jobs sequentially
for job in "${jobs[@]}"; do
  echo "Running: $job"
  eval "$job"
done
