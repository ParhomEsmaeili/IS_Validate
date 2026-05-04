#!/bin/bash
jobs=(
  "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task01_BrainTumour' -np=8"
  # "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task02_Heart' -np=8"
  "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task03_Liver' -np=8"
  "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task04_Hippocampus' -np=8"
  "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task05_Prostate' -np=8"
  "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task06_Lung' -np=8"
  "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task07_Pancreas' -np=8"
  "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task08_HepaticVessel' -np=8"
  # "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task09_Spleen' -np=8"
  "python3 convert_semantic_MSD_CL.py -d_path='/home/parhomesmaeili/Radiology_Datasets/MSD/Task10_Colon' -np=8"
)
# Run jobs sequentially
for job in "${jobs[@]}"; do
  echo "Running: $job"
  eval "$job"
done
