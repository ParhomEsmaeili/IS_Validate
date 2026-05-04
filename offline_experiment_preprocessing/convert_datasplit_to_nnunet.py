#This script is intended for converting semantic segmentation datasets from the format provided to that required by nnU-Net as this serves as one of the baseline
#comparisons for determining segmentation convergence. Only supports implementation for 

import argparse
import multiprocessing
import shutil
from typing import Optional
import SimpleITK as sitk
import os 
import sys 
import numpy as np
import json
import warnings
import copy
from skimage.measure import label as cc_label
from utils import (
    load_json, 
    save_json, 
    full_path_splitter, 
    file_ext_splitter,
)
  
def convert_data_split(
        target_dataset_dir_path:str,
        dataset_split_path:str,
        split_name: str,
        split_type: str,       
        dataset_name: str,
        ) -> None:
    target_folder = os.path.join(target_dataset_dir_path, dataset_name)
    # assert os.path.exists(target_folder), f"Target folder {target_folder} does not exist. Please provide a valid target folder path for the conversion process."
    os.makedirs(target_folder, exist_ok=True)

    if split_name == 'train':
        #Here we will convert the k-fold split into nnu-net format.
        assert split_type.endswith('train'), f"Unsupported split type {split_type} for split name 'train'. Only split types ending with 'train' are supported for split name 'train'. Got split type: {split_type}"
        existing_splits = load_json(dataset_split_path)['sampling']
        #Now lets extract the relevant split. 
        if split_type.startswith('kfold'):
            split_dict = existing_splits[split_type]
            reformatted_split = [] 
            for i in range(split_dict['meta']['k_folds']):
                current_split = dict()
                current_split['train'] = []
                current_split['val'] = split_dict[f'fold_{i}']
                for fold_j in range(split_dict['meta']['k_folds']):
                    if fold_j != i:
                        current_split['train'].extend(split_dict[f'fold_{fold_j}'])
                
                current_split['train'] = sorted(current_split['train'])
                current_split['val'] = sorted(current_split['val'])
                reformatted_split.append(current_split)
        
            
        else:
            raise ValueError(f"Unsupported split type {split_type}, only 'k_fold' is supported. For 'all-cases', use the default nnunet procedure for generating the split.")
    elif split_name == 'test':
        raise Exception("Data splitting for a hold-out test set is not relevant for nnu-net.")
    else:
        raise ValueError(f"Unsupported split name {split_name}, only 'train' and 'test' are supported.")

    output_json = reformatted_split
    save_json(output_json, os.path.join(target_folder, 'splits_final.json'), sort_keys=False, indent=4)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--reference_dataset_path', type=str, required=True, help='Path to the reference dataset which is to be used for processing.')
    argparser.add_argument('--target_dataset_basedir_path', type=str, required=True, help='Path to the target directory where the converted dataset will be stored.')
    argparser.add_argument('--split_name', type=str, default='train', help='Name of the split being processed')
    argparser.add_argument('--split_type', type=str, default='k_fold_5_train', help='Type of the split being processed, e.g., k_fold_5_train, all_cases_train, all_cases_test')
    
    args = argparser.parse_args()
    dataset_name = os.path.basename(os.path.dirname(args.reference_dataset_path))
    dataset_split_path = os.path.join(args.reference_dataset_path, 'dataset_split.json')
    convert_data_split(
        args.target_dataset_basedir_path, 
        dataset_split_path, 
        args.split_name, 
        args.split_type,
        dataset_name)