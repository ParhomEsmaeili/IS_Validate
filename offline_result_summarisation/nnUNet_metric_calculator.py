import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import argparse
import pandas as pd
from src.results_utils.base_metrics.scoring_base_utils import BaseScoringWrapper
from src.results_utils.metric_save_util import init_all_csvs, write_to_csvs
from src.general_utils.dict_utils import extractor
 
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
    for idx, file in enumerate(datalist):
        print(f'Calculating metrics for case: {file}, {idx+1}/{len(datalist)}')
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
    parser.add_argument('--segmentation_base_folder', type=str, required=True, help='Path to the base folder containing the dataset folders with segmentation and gold-standard annotations for metric calculation.')
    parser.add_argument('--gt_base_folder', type=str, required=True, help='Path to the base folder containing the dataset-separated folders which contain the gold standard annotations')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the task for reading segmentations, metric configs AND saving metrics.')
    parser.add_argument('--seg_subfolder_path', type=str, default='preds_final')
    parser.add_argument('--output_results_base_folder', type=str, default=os.path.join(parent_dir, 'results_summary'), help='Path to the base folder for saving the results summaries.')
    
    #For processing any specific experiment configs, or metric configs.
    parser.add_argument('--exp_configs_base_folder', type=str, default=os.path.join(parent_dir, 'exp_configs'), help='Path to the base folder containing experiment configs required for extracting metric and task configuration information')
    parser.add_argument('--reference_splits_base_folder', type=str, default=os.path.join(parent_dir, 'datasets'), help='Path to the base folder containing dataset folders with dataset_split.json files for reading the data splits.')
    parser.add_argument('--task_conf_id', type=str, default='task_id_2', help='Task configuration ID to use for reading the task configs which are to be used in loading reference labels, etc.')
    parser.add_argument('--metric_config_id', type=str, default='prototype', help='ID or string for the metric configuration for the given task')
    #Data extraction parameters required to identify the datalist!
    parser.add_argument('--data_split', type=str, default='test', choices=['train', 'test'], help='Name of the data split to be used for calculating metric')
    parser.add_argument('--strategy_type', type=str, required=False, choices=['all', 'kfold'], help='Sampling strategy type', default='all')
    parser.add_argument('--total_folds', type=int, default=None, help='If using kfold strategy, specify the total number of folds.')
    parser.add_argument('--fold_num', type=int, default=None, help='If using kfold strategy, specify the fold number (0-indexed) to use.')
    parser.add_argument('--gt_subfolder_path', type=str, default='labelsTs')
    # parser.add_argument('--metrics', nargs='+', type=str, default=['Dice', 'NSD'], help='List of metrics to calculate (default: Dice, NSD).')
    args = parser.parse_args()

    path_to_metric_configs = os.path.join(args.exp_configs_base_folder, args.dataset_name, 'metrics_configs.txt')
    path_to_task_configs = os.path.join(args.exp_configs_base_folder, args.dataset_name, 'task_configs.txt')
    output_folder = os.path.join(args.output_results_base_folder, args.dataset_name, 'nnUNet_metrics')
    
    seg_folder = os.path.join(
        args.segmentation_base_folder, 
        args.dataset_name, 
        args.seg_subfolder_path        
        )
    
    gt_folder = os.path.join(
        args.gt_base_folder, 
        args.dataset_name, 
        args.gt_subfolder_path)

    #Now we need to read in the datalist. 
    data_split_path = os.path.join(args.reference_splits_base_folder, args.dataset_name, 'dataset_split.json')

    if not os.path.exists(data_split_path):
        raise FileNotFoundError(f"Dataset split file not found at {data_split_path}")
    
    #Set the path to the correct split using the args.
    if args.strategy_type == 'all':
        dictionary_split_path = ('sampling', f'{args.strategy_type}_{args.data_split}', 'all_cases')
    elif args.strategy_type == 'kfold':
        if args.fold_num is None:
            raise ValueError('If using kfold strategy, please specify the fold number using --fold_num argument.')
        dictionary_split_path = ('sampling', f'{args.strategy_type}_{args.fold_num}_{args.data_split}', f'fold_{args.fold_num}')
    else:
        raise ValueError(f"Unknown strategy type: {args.strategy_type}")
    
    with open(data_split_path, 'r') as f:
        data_split = json.load(f)
        datalist = extractor(data_split, dictionary_split_path)
    


    #Read the metric configurations from the file
    if not os.path.exists(path_to_metric_configs):
        raise FileNotFoundError(f"Metric configuration file not found at {path_to_metric_configs}")
    else:
        with open(path_to_metric_configs, 'r') as f:
            configs_registry = json.load(f)
            metric_config = configs_registry[args.metric_config_id]

    if not os.path.exists(path_to_task_configs):
        raise FileNotFoundError(f"Task configuration file not found at {path_to_task_configs}")
    else:
        with open(path_to_task_configs, 'r') as f:
            task_configs = json.load(f)
            current_task = task_configs[args.task_conf_id] 
            config_labels_dict = current_task['data_transforms']['semantic_class_mapping']
            config_labels_dict = {k:idx for idx, k in enumerate(config_labels_dict.keys())}  # Convert to a dictionary with labels as keys and indices as values

    #Reading the config labels dictionary from the task definition.

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