import os
import sys
import copy
import logging
from datetime import datetime
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
import argparse
import pandas as pd
from src.general_utils.dict_utils import extractor
from surface_distance import compute_surface_distances, compute_robust_hausdorff 
from offline_result_summarisation.utils import extract_config, convert_nnunet_task_to_internal_convention
from src.version_handling import monai_version
from monai.data import MetaTensor
import json 
import torch 
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
)

monai_transforms = Compose([
    LoadImaged(keys=['seg', 'gt'], image_only=True, reader='ITKReader'),  # Keep metadata for spacing extraction
    EnsureChannelFirstd(keys=['seg', 'gt']),
    Orientationd(keys=['seg', 'gt'], axcodes='RAS'),
    EnsureTyped(keys=['seg', 'gt'], dtype=[torch.uint8, torch.uint8]),
])

# Setup progress logging
def setup_logging(output_folder):
    log_file = os.path.join(output_folder, f'surface_distance_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def convert_numpy_to_python(obj):
    """
    Recursively convert numpy arrays and numpy scalar types to native Python types.
    Enables JSON serialization of results dictionaries.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def convert_tensor_to_numpy(tensor_data):
    """
    Convert MetaTensor to numpy array and extract image spacing.
    REQUIRES: Input must be a MetaTensor with affine metadata (for physical spacing).
    
    Args:
        tensor_data: Must be a MetaTensor with affine metadata
    
    Returns:
        tuple: (numpy_array, image_spacing) where spacing is guaranteed to be non-None
    
    Raises:
        TypeError if not a MetaTensor
        ValueError if spacing cannot be extracted from metadata
    """
    if monai_version != '1.4.0':
        raise Exception(f'This script requires MONAI version 1.4.0, but found {monai_version}. Please ensure you are using MONAI 1.4.0.')
    
    if not isinstance(tensor_data, MetaTensor):
        raise TypeError(f"Input must be a MetaTensor with affine metadata for spacing extraction. Got {type(tensor_data)}")
    
    array = tensor_data.array
    
    # Extract spacing from affine metadata - THIS IS MANDATORY
    if not hasattr(tensor_data, 'meta') or 'affine' not in tensor_data.meta:
        raise ValueError("MetaTensor must have 'affine' in metadata for imaging spacing extraction")
    
    val_dom_affine = copy.deepcopy(tensor_data.meta['affine'])
    if val_dom_affine is None:
        raise ValueError("Affine metadata is None, cannot extract imaging spacing")
    
    if val_dom_affine.shape[0] != 4:
        raise ValueError(f"Expected 4x4 affine matrix, got shape {val_dom_affine.shape}")
    
    dim = val_dom_affine.shape[0] - 1
    _m_key = (slice(-1), slice(-1))
    spacing = np.linalg.norm(val_dom_affine[_m_key] @ np.eye(dim), axis=0)
    
    if spacing is None or len(spacing) == 0:
        raise ValueError("Failed to extract imaging spacing from affine matrix")
    
    return array, spacing

def calculate_surface_distances(seg_folder, gt_folder, config_labels_dict, datalist, logger, percentile_for_hausdorff=95):
    """Calculate surface distances for all cases and all classes (excluding background)."""
    # Check all files exist
    for file in datalist:
        if not os.path.exists(os.path.join(gt_folder, file + '.nii.gz')):
            raise FileNotFoundError(f"Ground truth file not found for {file} in {gt_folder}")
        if not os.path.exists(os.path.join(seg_folder, file + '.nii.gz')):
            raise FileNotFoundError(f"Segmentation file not found for {file} in {seg_folder}")
    
    logger.info(f"Computing surface distances for {len(datalist)} cases and {len(config_labels_dict)} classes (excluding background)")
    
    surface_distances_dict = {}
    
    for idx, file in enumerate(datalist):
        logger.info(f'Processing case: {file}, {idx+1}/{len(datalist)}')
        
        seg_file_path = os.path.join(seg_folder, file + '.nii.gz')
        gt_file_path = os.path.join(gt_folder, file + '.nii.gz')

        # Load files using MONAI transforms (preserving metadata)
        data = monai_transforms({'seg': seg_file_path, 'gt': gt_file_path})
        
        seg_tensor = data['seg'][0]  # First axis is channel dim
        gt_tensor = data['gt'][0]
        
        # Convert to numpy and extract spacing from metadata (spacing is mandatory)
        seg_array, seg_spacing = convert_tensor_to_numpy(seg_tensor)
        gt_array, im_spacing = convert_tensor_to_numpy(gt_tensor)
        assert np.isclose(seg_spacing, im_spacing).all(), f"Spacing mismatch between seg and gt for case {file}: seg spacing {seg_spacing}, gt spacing {im_spacing}"
        seg_array = seg_array.astype(np.uint8)
        gt_array = gt_array.astype(np.uint8)
        
        logger.info(f'  Seg shape: {seg_array.shape}, GT shape: {gt_array.shape}, Image spacing: {im_spacing}')
        
        # Compute surface distances per class (excluding background/class 0)
        case_results = {}
        for class_name, class_idx in config_labels_dict.items():
            # Skip background class (typically class_idx == 0)
            if class_idx == 0:
                logger.info(f'  Skipping background class: {class_name}')
                continue
            
            logger.info(f'  Computing surface distances for class: {class_name} (idx={class_idx})')
            #we need to convert the seg array and gt array to bool types for the given class! it is
            #only compatible with binary sem seg.
            seg_input = (seg_array == class_idx).astype(bool)
            gt_input = (gt_array == class_idx).astype(bool)
            surface_dists = compute_surface_distances(seg_input, gt_input, spacing_mm=im_spacing)
            hd_dist = compute_robust_hausdorff(surface_dists, percent=percentile_for_hausdorff)
            surface_dists[f'hausdorff_distance_{percentile_for_hausdorff}'] = hd_dist
            # Log surface distance summary
            logger.info(f'    Hausdorff distance: {surface_dists[f"hausdorff_distance_{percentile_for_hausdorff}"]:.4f}')
            logger.info(f'    GT->Seg mean distance: {surface_dists["distances_gt_to_pred"].mean():.4f}')
            logger.info(f'    Seg->GT mean distance: {surface_dists["distances_pred_to_gt"].mean():.4f}')
            
            # Store result
            serializable_result = {k: v for k, v in surface_dists.items() if not k.startswith('_')}
            case_results[class_name] = serializable_result
        
        surface_distances_dict[file] = case_results
        
        # DEBUG: Space for custom analysis on raw surface distances
        # You can add custom debugging/analysis here with access to:
        # - surface_dists['_raw_dist_gt_to_seg']: distances from GT surface to seg
        # - surface_dists['_raw_dist_seg_to_gt']: distances from seg surface to GT
        # - surface_dists['_seg_surface_mask']: binary mask of seg surface
        # - surface_dists['_gt_surface_mask']: binary mask of GT surface
        # - seg_array: full segmentation array
        # - gt_array: full ground truth array
        
    return surface_distances_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate surface distances for nnUNet segmentations.")
    parser.add_argument(
        '--segmentation_base_folder', 
        type=str, 
        required=True, 
        # default='/home/parhomesmaeili/Helmholtz Group/nnUNet_inspect',
        help='Path to the base folder containing dataset folders with segmentations.'
        )
    parser.add_argument(
        '--gt_base_folder', 
        type=str, 
        required=True,
        # default='/home/parhomesmaeili/Helmholtz Group/nnUNet/nnUNet_raw',
        help='Path to the base folder containing dataset-separated folders with gold standard annotations.'
        )
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True,
        # default='Dataset046_MSMultispineAll',
        help='Name of the task in nnu-net convention, which is also the name of its dataset folder.'
        )
    parser.add_argument(
        '--seg_subfolder_path', 
        type=str, 
        required=True,
        # default='nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessed', #'preds_final',
        help='Subfolder name for segmentations.'
        )
    parser.add_argument(
        '--output_results_base_folder', 
        type=str, 
        default=os.path.join(parent_dir, 'results_summary'), 
        help='Path to the base folder for saving the results.'
        )
    #For processing any specific experiment configs.
    parser.add_argument(
        '--nnUNet_exp_configs_base_folder', 
        type=str, 
        default=os.path.join(parent_dir, 'exp_configs_nnUNet'), 
        help='Path to the base folder containing experiment configs required for extracting task configuration information.'
        )
    parser.add_argument(
        '--reference_splits_base_folder', 
        type=str, 
        default=os.path.join(parent_dir, 'datasets'), 
        help='Path to the base folder containing dataset folders with dataset_split.json files for reading the data splits.'
        )
    parser.add_argument(
        '--task_conf_id', 
        nargs='+', 
        type=str, 
        default=['task_id_3', 'task_id_4', 'task_id_5', 'task_id_6', 'task_id_7'], #['task_id_2'],
        help='Task configuration ID to use for reading the task configs which are to be used in loading reference labels, etc.'
        )
    #Data extraction parameters required to identify the datalist!
    parser.add_argument(
        '--data_split', 
        type=str, 
        default='train',
        choices=['train', 'test'], 
        help='Data split to use for surface distance calculation.'
        )
    parser.add_argument(
        '--strategy_type', 
        type=str, 
        required=False, #True,
        default='kfold',
        choices=['all', 'kfold'], 
        help='Sampling strategy type.'
        )
    parser.add_argument(
        '--total_folds', 
        type=int, 
        default=None, 
        help='If using kfold strategy, specify the total number of folds.'
        )
    parser.add_argument(
        '--gt_subfolder_path', 
        type=str, 
        default='labelsTr',
        help='Subfolder name for ground truth annotations.'
        )
    parser.add_argument(
        '--HD_percentile',
        type=float,
        default=95.0,
        help='Percentile to use for Hausdorff distance calculation (default: 95).'
    )
    args = parser.parse_args()

    mapping_configs = [convert_nnunet_task_to_internal_convention(
        nnunet_task_id=conf_id,
        mapping_file_path=os.path.join(args.nnUNet_exp_configs_base_folder, args.dataset_name, 'task_configs.json')
    )
    for conf_id in args.task_conf_id
    ]
    nnunet_dataset_name = args.dataset_name 
    del args.dataset_name #We will remove this to prevent it leaking. 
    
    assert len(set([mapping['dataset_name_val_convention'] for mapping in mapping_configs])) == 1, 'All provided task_conf_id values must correspond to the same dataset_name_val_convention in the mapping configs, please check your task configs and mapping configs to ensure this is the case.'
    assert len(set([mapping['exp_config_relpath'] for mapping in mapping_configs])) == 1, 'All provided task_conf_id values must correspond to the same exp_config_relpath in the mapping configs, please check your task configs and mapping configs to ensure this is the case.'
    
    val_framework_dataset_name = mapping_configs[0]['dataset_name_val_convention']
    exp_config_relpath = mapping_configs[0]['exp_config_relpath']
    
    print(f"Dataset name in original convention from mapping configs: {val_framework_dataset_name}")
    print(f"Experiment config relative path from mapping configs: {exp_config_relpath}")

    output_folder = os.path.join(args.output_results_base_folder, nnunet_dataset_name, 'nnUNet_surface_distances')
    logger = setup_logging(output_folder)
    
    # Path to task configs for loading semantic_class_mapping
    path_to_task_configs = os.path.join(parent_dir, exp_config_relpath, "task_configs.txt")
    
    logger.info(f"Starting surface distance calculation for dataset: {nnunet_dataset_name}")
    logger.info(f"Segmentation folder: {os.path.join(args.segmentation_base_folder, nnunet_dataset_name, args.seg_subfolder_path)}")
    logger.info(f"Ground truth folder: {os.path.join(args.gt_base_folder, nnunet_dataset_name, args.gt_subfolder_path)}")
    
    seg_folder = os.path.join(args.segmentation_base_folder, nnunet_dataset_name, args.seg_subfolder_path)
    gt_folder = os.path.join(args.gt_base_folder, nnunet_dataset_name, args.gt_subfolder_path)

    # Load semantic class mapping from task configs
    if not os.path.exists(path_to_task_configs):
        raise FileNotFoundError(f"Task configuration file not found at {path_to_task_configs}")
    
    current_tasks = [extract_config(path_to_task_configs, conf_id) for conf_id in args.task_conf_id]
    
    # Extract semantic class mapping from task configs
    semantic_class_mappings = [task['data_transforms']['semantic_class_mapping'] for task in current_tasks]
    canonical_mappings = [json.dumps(m, sort_keys=True) for m in semantic_class_mappings]
    if len(set(canonical_mappings)) != 1:
        raise ValueError(f"Semantic class mappings are not the same across the tasks specified by task_conf_id {args.task_conf_id}. Please ensure they are the same, or specify a single task_conf_id corresponding to a single task configuration.")
    
    config_labels_dict = semantic_class_mappings[0]
    config_labels_dict = {k: idx for idx, k in enumerate(config_labels_dict.keys())}  # Convert to {label_name: index}
    logger.info(f"Config labels dict: {config_labels_dict}")

    # Read data split
    data_split_path = os.path.join(args.reference_splits_base_folder, val_framework_dataset_name, 'dataset_split.json')
    if not os.path.exists(data_split_path):
        raise FileNotFoundError(f"Dataset split file not found at {data_split_path}")
    
    if args.strategy_type == 'all':
        dictionary_split_paths = [('sampling', f'{args.strategy_type}_{args.data_split}', 'all_cases')]
    elif args.strategy_type == 'kfold':
        if args.total_folds is None:
            raise ValueError('If using kfold strategy, please specify --total_folds.')
        dictionary_split_paths = [('sampling', f'{args.strategy_type}_{args.total_folds}_{args.data_split}', f'fold_{fold_num}') for fold_num in range(args.total_folds)]
    else:
        raise ValueError(f"Unknown strategy type: {args.strategy_type}")
    
    with open(data_split_path, 'r') as f:
        data_split = json.load(f)
        logger.info(f"Data split json loaded")
        print(f"Dictionary split paths: \n{dictionary_split_paths}")
        datalist = []
        for path in dictionary_split_paths:
            datalist.extend(extractor(data_split, path))
    
    logger.info(f"Number of cases in datalist: {len(datalist)}")
    
    # Calculate surface distances
    surface_distances_dict = calculate_surface_distances(
        seg_folder, 
        gt_folder, 
        config_labels_dict,
        datalist,
        logger,
        percentile_for_hausdorff=args.HD_percentile
    )
    
    # Save results as JSON (convert numpy types to native Python types for JSON serialization)
    output_json = os.path.join(output_folder, 'surface_distances.json')
    os.makedirs(output_folder, exist_ok=True)
    serializable_results = convert_numpy_to_python(surface_distances_dict)
    with open(output_json, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved surface distance results to {output_json}")
    logger.info(f"Completed surface distance calculation for {len(surface_distances_dict)} cases")
