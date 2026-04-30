import os
import sys
import copy
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import argparse
import pandas as pd
from src.results_utils.base_metrics.scoring_base_utils import BaseScoringWrapper
from src.results_utils.metric_save_util import init_all_csvs, write_to_csvs
from src.general_utils.dict_utils import extractor
from offline_result_summarisation.utils import extract_config, convert_nnunet_task_to_internal_convention
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

def process_metric_config(metric_config, spacing_config):
    #Function which processes the metric configs for cases where we do not have a trivial config.
    #E.g., for NSD where we may want to calculate across multiple tolerance values.
    for metric_name, conf in metric_config.items():
        if metric_name == 'NSD':
            if 'tolerance_mm' not in conf:
                assert 'tolerance_sf' in conf, 'If no tolerance_mm provided for NSD metric config, then must provide a tolerance_sf value to calculate the tolerance based on the dataset voxel spacing, please check your metric configs.'
                #If we have a tolerance sf, then we must calculate tolerance mms from the tolerance sf. 
                if spacing_config:
                    tolerance_mms = [float(i) * float(spacing_config) for i in conf['tolerance_sf']]
                    assert isinstance(tolerance_mms, list), 'The calculated tolerance mms values must be a list, even if there is only one value, to be consistent with the case where the tolerance mms values are provided directly in the metric configs, please check your metric configs and dataset spacing config to ensure this is the case.'
                    conf['tolerance_mms'] = {index:tolerance_mm for index,tolerance_mm in enumerate(tolerance_mms)}

                    #We add the tolerance mms values to the config dict for the metric, we will use these for the metric calculations. We deepcopy just to be safe and avoid any weird pointer issues.
                    del conf['tolerance_sf'] #We remove the tolerance sf value as it is no longer needed and to avoid confusion.
                else:
                    raise ValueError('If using a tolerance sf value for NSD metric config, then the dataset spacing config must be provided to calculate the tolerance_mm values, please check your metric configs and dataset configs to ensure this is the case.')
            else:
                tolerance_mms = copy.deepcopy(conf['tolerance_mm'])
                del conf['tolerance_mm'] #We remove the tolerance mm value as it is no longer needed and to avoid confusion, we will just use the tolerance mms values for the metric calculations. We deepcopy just to be safe and avoid any weird pointer issues.
                conf['tolerance_mms'] = {0: tolerance_mms} #We just rename the key to be consistent with the case where we calculate the tolerance mms from the tolerance sf, this is just
            if len(conf['tolerance_mms']) > 1:
                #We add a new key which says that we have multiple nsds so downstream it knows to treat
                #this differently!. It points to the key where that corresponding list of parameterisations
                #are.
                conf['multiple_parameter_values'] = 'tolerance_mms'
    return metric_config

def write_metrics(metrics_dict, metrics_configs, config_labels_dict, output_base_folder):
    metric_paths_dict = init_all_csvs(output_base_folder, metrics_configs, config_labels_dict)

    for case, metrics in metrics_dict.items():
        write_to_csvs(
            case_name=case,
            csv_paths=metric_paths_dict,
            tracked_metrics=metrics,
            metrics_configs=metrics_configs,
        )
    print('done writing metrics.')

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
    parser.add_argument('--segmentation_base_folder', 
        type=str, 
        required=True,
        default='/home/parhomesmaeili/Helmholtz Group/nnUNet_inspect',
        help='Path to the base folder containing the dataset folders with segmentation and gold-standard annotations for metric calculation.')
    parser.add_argument('--gt_base_folder', 
        type=str, 
        required=True,
        default='/home/parhomesmaeili/Helmholtz Group/nnUNet/nnUNet_raw',
        help='Path to the base folder containing the dataset-separated folders which contain the gold standard annotations')
    parser.add_argument('--dataset_name', 
        type=str, 
        required=True,
        default='Dataset046_MSMultispineAll',
        help='Name of the task in nnu-net convention, which is also the name of its dataset folder.')
    parser.add_argument('--seg_subfolder_path', 
        type=str, 
        default="nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessed")#'preds_final')
    parser.add_argument('--output_results_base_folder', 
        type=str, 
        default=os.path.join(parent_dir, 'results_summary'), 
        help='Path to the base folder for saving the results summaries.')
    #For processing any specific experiment configs, or metric configs.
    parser.add_argument('--nnUNet_exp_configs_base_folder', 
        type=str, 
        default=os.path.join(parent_dir, 'exp_configs_nnUNet'), 
        help='Path to the base folder containing experiment configs required for extracting metric and task configuration information')
    parser.add_argument('--reference_splits_base_folder', 
        type=str, 
        default=os.path.join(parent_dir, 'datasets'), 
        help='Path to the base folder containing dataset folders with dataset_split.json files for reading the data splits.')
    parser.add_argument('--task_conf_id', 
        nargs='+', 
        type=str, 
        default=['task_id_3', 'task_id_4', 'task_id_5', 'task_id_6', 'task_id_7'], #'task_id_2', 
        help='Task configuration ID to use for reading the task configs which are to be used in loading reference labels, etc.')
    parser.add_argument('--metric_config_id', 
        type=str, 
        default='prototype', 
        help='ID or string for the metric configuration for the given task')
    #Data extraction parameters required to identify the datalist!
    parser.add_argument('--data_split', 
        type=str, 
        default='train', #'test', 
        choices=['train', 'test'], 
        help='Name of the data split to be used for calculating metric')
    parser.add_argument('--strategy_type', 
        type=str, 
        required=True,
        choices=['all', 'kfold'], 
        default='kfold',
        help='Sampling strategy type')
    parser.add_argument('--total_folds', 
        type=int, 
        default=None, 
        help='If using kfold strategy, specify the total number of folds.')
    # parser.add_argument('--fold_num', type=int, default=None, help='If using kfold strategy, specify the fold number (0-indexed) to use.')
    parser.add_argument('--gt_subfolder_path', 
        type=str, 
        default='labelsTr'
    )
    # parser.add_argument('--metrics', nargs='+', type=str, default=['Dice', 'NSD'], help='List of metrics to calculate (default: Dice, NSD).')
    args = parser.parse_args()

    mapping_configs = [convert_nnunet_task_to_internal_convention(
        nnunet_task_id=conf_id,
        mapping_file_path=os.path.join(args.nnUNet_exp_configs_base_folder, args.dataset_name, 'task_configs.json')
    )
    for conf_id in args.task_conf_id
    ]
    
    assert len(set([mapping['dataset_name_val_convention'] for mapping in mapping_configs])) == 1, 'All provided task_conf_id values must correspond to the same dataset_name_val_convention in the mapping configs, please check your task configs and mapping configs to ensure this is the case.'
    assert len(set([mapping['exp_config_relpath'] for mapping in mapping_configs])) == 1, 'All provided task_conf_id values must correspond to the same exp_config_relpath in the mapping configs, please check your task configs and mapping configs to ensure this is the case.'
    #We must be accessing the same dataset and task_relpath across all provided task_conf_id values, as these correspond to the dataset and task_relpath which we will use for accessing the datalist and loading the reference labels for metric calculation, so we need to ensure consistency here. If there are multiple task_conf_id values provided, then they should only differ in terms of the specific data sampling (e.g., different folds in a kfold strategy) or data transformations (e.g., different permutations of the segmentations), but they should all point to the same dataset and task_relpath for accessing the datalist and loading reference labels, as these are not
    #things that we want to differ across different task IDs within a dataset in nnUNet (here the IDs
    #are typically just different folds).
    val_framework_dataset_name = mapping_configs[0]['dataset_name_val_convention']
    exp_config_relpath = mapping_configs[0]['exp_config_relpath']
    
    nnunet_dataset_name = args.dataset_name 
    del args.dataset_name #We will remove this to prevent it leaking. 
    
    print(f"Dataset name in original convention from mapping configs: {val_framework_dataset_name}")
    print(f"Experiment config relative path from mapping configs: {exp_config_relpath}")

    # path_to_metric_configs = os.path.join(args.exp_configs_base_folder, args.dataset_name, 'metrics_configs.txt')
    # path_to_task_configs = os.path.join(args.exp_configs_base_folder, args.dataset_name, 'task_configs.txt')
    # output_folder = os.path.join(args.output_results_base_folder, args.dataset_name, 'nnUNet_metrics')
    # spacing_config_path = os.path.join(args.exp_configs_base_folder, args.dataset_name, 'spacing_config.json')
    path_to_metric_configs = os.path.join(parent_dir, exp_config_relpath, "metrics_configs.txt")
    path_to_task_configs = os.path.join(parent_dir, exp_config_relpath, "task_configs.txt")
    output_folder = os.path.join(args.output_results_base_folder, nnunet_dataset_name, 'nnUNet_metrics')
    spacing_config_path = os.path.join(parent_dir, exp_config_relpath, 'spacing_config.json')

    seg_folder = os.path.join(
        args.segmentation_base_folder, 
        nnunet_dataset_name, 
        args.seg_subfolder_path        
        )
    
    gt_folder = os.path.join(
        args.gt_base_folder, 
        nnunet_dataset_name, 
        args.gt_subfolder_path)

    #Now we need to read in the datalist. 
    data_split_path = os.path.join(args.reference_splits_base_folder, val_framework_dataset_name, 'dataset_split.json')

    if not os.path.exists(data_split_path):
        raise FileNotFoundError(f"Dataset split file not found at {data_split_path}")
    
    #Set the path to the correct split using the args.
    if args.strategy_type == 'all':
        dictionary_split_paths = [('sampling', f'{args.strategy_type}_{args.data_split}', 'all_cases')]
    elif args.strategy_type == 'kfold':
        if args.total_folds is None:
            raise ValueError('If using kfold strategy, please specify the number of folds using --total_folds argument.')
        dictionary_split_paths = [('sampling', f'{args.strategy_type}_{args.total_folds}_{args.data_split}', f'fold_{fold_num}') for fold_num in range(args.total_folds)]
    else:
        raise ValueError(f"Unknown strategy type: {args.strategy_type}")
    
    with open(data_split_path, 'r') as f:
        data_split = json.load(f)
        print(f'Data split json dict: {data_split}')
        print(f"Dictionary split paths: \n{dictionary_split_paths}")
        datalist = []
        for path in dictionary_split_paths:
            datalist.extend(extractor(data_split, path))
    print(f"Number of cases in the datalist: {len(datalist)}")


    #Read the metric configurations from the file
    if not os.path.exists(path_to_metric_configs):
        raise FileNotFoundError(f"Metric configuration file not found at {path_to_metric_configs}")
    else:
        metric_config = extract_config(path_to_metric_configs, args.metric_config_id) #We also do this as a check to ensure the metric config id provided is valid, this will raise an error if it is not. If it is valid, then we will read in the metric configs again in the next step when we initialise the scoring wrapper, so we don't need to keep the metric config dict returned from this function call.
        # with open(path_to_metric_configs, 'r') as f:
        #     configs_registry = json.load(f)
        #     metric_config = configs_registry[args.metric_config_id]

    if not os.path.exists(path_to_task_configs):
        raise FileNotFoundError(f"Task configuration file not found at {path_to_task_configs}")
    else:
        current_tasks = [extract_config(path_to_task_configs, conf_id) for conf_id in args.task_conf_id]
        # with open(path_to_task_configs, 'r') as f:
        #     task_configs = json.load(f)
        #     current_tasks = [task_configs[conf_id] for conf_id in args.task_conf_id]
        #We assert uniqueness of the semantic class mapping.
        
        for fold_idx, task in enumerate(current_tasks):
            if 'semantic_class_mapping' not in task['data_transforms']:
                raise KeyError(f"Task configuration with ID {args.task_conf_id} does not contain 'semantic_class_mapping' in its 'data_transforms' section.")
            if f'fold_{fold_idx}' not in task['data_sampling']['sample_group_category']:
                raise KeyError(f"Task configuration with ID {args.task_conf_id} does not contain fold-specific semantic class mapping for fold_{fold_idx}")
            
        semantic_class_mappings = [task['data_transforms']['semantic_class_mapping'] for task in current_tasks]
        canonical_mappings = [json.dumps(m, sort_keys=True) for m in semantic_class_mappings]
        if len(set(canonical_mappings)) != 1:
            raise ValueError(f"Semantic class mappings are not the same across the tasks specified by task_conf_id {args.task_conf_id}. Please ensure they are the same, or specify a single task_conf_id corresponding to a single task configuration.")
        else:
            config_labels_dict = semantic_class_mappings[0]
        print(f"Config labels dict: {config_labels_dict}")
        # config_labels_dict = current_tasks[0]['data_transforms']['semantic_class_mapping']
        config_labels_dict = {k:idx for idx, k in enumerate(config_labels_dict.keys())}  # Convert to a dictionary with labels as keys and indices as values

    #Lets extract the spacing config if it exists. We will use a try-except here for convenience.
    try:
        spacing_config = extract_config(spacing_config_path, 'reference_spacing')
    except:
        spacing_config = None 
    
    #Here we will process the metric configs, as there may be instances where we have multiple parameter
    #values for a given metric... (For example with NSD if we do not have a single tolerance provided).
    metric_config = process_metric_config(
        metric_config=metric_config,
        spacing_config=spacing_config
    )

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
