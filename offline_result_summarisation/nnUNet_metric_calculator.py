import os
import sys 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import argparse
import pandas as pd
from src.results_utils.base_metrics.scoring_base_utils import BaseScoringWrapper
from src.results_utils.metric_save_util import init_all_csvs, write_to_csvs
 
import json 
import torch 
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
)

monai_transforms = Compose([
    LoadImaged(keys=['seg', 'gt'], image_only=True),
    EnsureChannelFirstd(keys=['seg', 'gt']),
    Orientationd(keys=['seg', 'gt'], axcodes='RAS'),
    EnsureTyped(keys=['seg', 'gt'], dtype=[torch.uint8, torch.uint8]),
])

def write_metrics(metrics_dict, metrics_configs, config_labels_dict, output_base_folder):
    metric_paths_dict = init_all_csvs(output_base_folder, metrics_configs, config_labels_dict)

    for case, metrics in metrics_dict.items():
        write_to_csvs(
            case_name=case,
            csv_paths=metric_paths_dict,
            tracked_metrics=metrics
        )
    print('done')

def calculate_nnUNet_metrics(scoring_wrapper, seg_folder, gt_folder, config_labels_dict, datalist):
    #check all of the files are in the seg and gt folder.
    for file in datalist:
        if not os.path.exists(os.path.join(gt_folder, file + '.nii.gz')):
            raise FileNotFoundError(f"Ground truth file not found for {file} in {gt_folder}")
        if not os.path.exists(os.path.join(seg_folder, file + '.nii.gz')):
            raise FileNotFoundError(f"Segmentation file not found for {file} in {seg_folder}")
        

    #now calculate the metrics for each file.
    metrics_dict = {}
    for file in datalist:
        seg_file_path = os.path.join(seg_folder, file + '.nii.gz')
        gt_file_path = os.path.join(gt_folder, file + '.nii.gz')

        #Load the segmentation and ground truth files using MONAI transforms
        data = monai_transforms({'seg': seg_file_path, 'gt': gt_file_path})
        
        seg_tensor = data['seg'][0] #Assuming the first axis is the channel dim.
        gt_tensor = data['gt'][0] #Assuming the first axis is the channel dim.

        # Calculate the metrics using the scoring wrapper
        img_masks = (
            torch.ones_like(seg_tensor, dtype=torch.int64), #'cross_class_mask':
            {class_lab:torch.ones_like(seg_tensor,dtype=torch.int64) for class_lab in config_labels_dict.keys()} #'per_class_masks': 
        )

        metrics = scoring_wrapper(
            image_masks=img_masks,
            pred=seg_tensor,
            gt=gt_tensor
        )

        metrics_dict[file] = {
            k: {'Automatic Init': v} for k,v in metrics.items()
        }
    
    return metrics_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate nnUNet metrics.")
    # required=True, help='Path to the input folder containing results.')
    parser.add_argument('--dataset_name', type=str, default='Dataset007_Pancreas', help='Name of task for saving metrics.')
    parser.add_argument('--seg_subfolder_path', type=str, default='postprocessed')
    # parser.add_argument('--metrics', nargs='+', type=str, default=['Dice', 'NSD'], help='List of metrics to calculate (default: Dice, NSD).')
    args = parser.parse_args()

    path_to_metric_configs = os.path.join(parent_dir, 'exp_configs', args.dataset_name, 'metrics_configs.txt')
    path_to_task_configs = os.path.join(parent_dir, 'exp_configs', args.dataset_name, 'task_configs.txt')
    output_folder = os.path.join(parent_dir, 'results_summary', args.dataset_name, 'nnUNet_metrics')
    
    seg_folder = os.path.join(
        '/home/parhomesmaeili/Helmholtz Group/MICCAI2025_nnunet/nnUNet_results/pretzel_best_config_crossval', 
        args.dataset_name, 
        args.seg_subfolder_path        
        )
    
    gt_folder = os.path.join(
        '/home/parhomesmaeili/Helmholtz Group/MICCAI2025_nnunet/nnUNet_raw', 
        args.dataset_name, 
        'labelsTr')

    #Read the metric configurations from the file
    if not os.path.exists(path_to_metric_configs):
        raise FileNotFoundError(f"Metric configuration file not found at {path_to_metric_configs}")
    else:
        with open(path_to_metric_configs, 'r') as f:
            configs_registry = json.load(f)
            metric_config = configs_registry['prototype']

    if not os.path.exists(path_to_task_configs):
        raise FileNotFoundError(f"Task configuration file not found at {path_to_task_configs}")
    else:
        with open(path_to_task_configs, 'r') as f:
            task_configs = json.load(f)
            current_task = task_configs['task_id_1'] 
            config_labels_dict = current_task['data_transforms']['semantic_class_mapping']
            config_labels_dict = {k:idx for idx, k in enumerate(config_labels_dict.keys())}  # Convert to a dictionary with labels as keys and indices as values

    #Reading the config labels dictionary from the task definition.

    #Now we need to read in the datalist. 
    data_split_path = os.path.join(parent_dir, 'datasets', args.dataset_name, 'dataset_split.json')

    if not os.path.exists(data_split_path):
        raise FileNotFoundError(f"Dataset split file not found at {data_split_path}")
    with open(data_split_path, 'r') as f:
        data_split = json.load(f)
        datalist = data_split['sampling']['all_train']['all_cases']
    
    # Initialise the scoring wrapper
    scoring_wrapper = BaseScoringWrapper(
        calc_device=torch.device(0),
        metrics_configs=metric_config, 
        config_labels_dict=config_labels_dict,
    )

    # Calculate nnUNet metrics
    metrics_dict = calculate_nnUNet_metrics(
        scoring_wrapper, 
        seg_folder, 
        gt_folder, 
        config_labels_dict, 
        datalist)

    #write the metrics to a csv.  
    write_metrics(metrics_dict, metric_config, config_labels_dict, output_folder)