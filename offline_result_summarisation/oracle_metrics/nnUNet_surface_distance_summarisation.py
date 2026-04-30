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
from scipy.ndimage import distance_transform_edt

monai_transforms = Compose([
    LoadImaged(keys=['seg', 'gt'], image_only=True),
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

def extract_config(path, name):
    """Function which extracts configs dicts from json or txt files."""
    if not os.path.exists(path):
        raise Exception(f'The path {path} was not a valid one. Please check.')    
    with open(path) as f:
        configs_registry = json.load(f)
        config = configs_registry[name]
    return config 

def compute_surface_distances(seg_array, gt_array, spacing=None):
    """
    Compute surface distances between segmentation and ground truth.
    
    Args:
        seg_array: segmentation array (numpy)
        gt_array: ground truth array (numpy)
        spacing: voxel spacing (for physical distance calculation)
    
    Returns:
        dict with surface distance information
    """
    # Ensure binary arrays
    seg_binary = (seg_array > 0).astype(np.uint8)
    gt_binary = (gt_array > 0).astype(np.uint8)
    
    # Compute distance transforms
    # Distance from pred surface to gt
    seg_dist = distance_transform_edt(~seg_binary)
    gt_dist = distance_transform_edt(~gt_binary)
    
    # Get surface points (boundaries)
    seg_surface = (seg_dist == 0) & seg_binary
    gt_surface = (gt_dist == 0) & gt_binary
    
    # Distance from gt surface to pred
    if np.any(gt_surface):
        dist_gt_to_seg = seg_dist[gt_surface]
    else:
        dist_gt_to_seg = np.array([])
    
    # Distance from pred surface to gt
    if np.any(seg_surface):
        dist_seg_to_gt = gt_dist[seg_surface]
    else:
        dist_seg_to_gt = np.array([])
    
    # Compute statistics
    result = {
        'num_seg_surface_voxels': int(np.sum(seg_surface)),
        'num_gt_surface_voxels': int(np.sum(gt_surface)),
    }
    
    if len(dist_gt_to_seg) > 0:
        result['dist_gt_to_seg_mean'] = float(np.mean(dist_gt_to_seg))
        result['dist_gt_to_seg_median'] = float(np.median(dist_gt_to_seg))
        result['dist_gt_to_seg_max'] = float(np.max(dist_gt_to_seg))
        result['dist_gt_to_seg_std'] = float(np.std(dist_gt_to_seg))
    else:
        result['dist_gt_to_seg_mean'] = np.nan
        result['dist_gt_to_seg_median'] = np.nan
        result['dist_gt_to_seg_max'] = np.nan
        result['dist_gt_to_seg_std'] = np.nan
    
    if len(dist_seg_to_gt) > 0:
        result['dist_seg_to_gt_mean'] = float(np.mean(dist_seg_to_gt))
        result['dist_seg_to_gt_median'] = float(np.median(dist_seg_to_gt))
        result['dist_seg_to_gt_max'] = float(np.max(dist_seg_to_gt))
        result['dist_seg_to_gt_std'] = float(np.std(dist_seg_to_gt))
    else:
        result['dist_seg_to_gt_mean'] = np.nan
        result['dist_seg_to_gt_median'] = np.nan
        result['dist_seg_to_gt_max'] = np.nan
        result['dist_seg_to_gt_std'] = np.nan
    
    # Hausdorff distance (symmetric)
    if len(dist_gt_to_seg) > 0 and len(dist_seg_to_gt) > 0:
        result['hausdorff_distance'] = float(max(np.max(dist_gt_to_seg), np.max(dist_seg_to_gt)))
    else:
        result['hausdorff_distance'] = np.nan
    
    # DEBUG: Raw distance arrays available for further analysis
    result['_raw_dist_gt_to_seg'] = dist_gt_to_seg
    result['_raw_dist_seg_to_gt'] = dist_seg_to_gt
    result['_seg_surface_mask'] = seg_surface
    result['_gt_surface_mask'] = gt_surface
    
    return result

def calculate_surface_distances(seg_folder, gt_folder, config_labels_dict, datalist, logger):
    """Calculate surface distances for all cases."""
    # Check all files exist
    for file in datalist:
        if not os.path.exists(os.path.join(gt_folder, file + '.nii.gz')):
            raise FileNotFoundError(f"Ground truth file not found for {file} in {gt_folder}")
        if not os.path.exists(os.path.join(seg_folder, file + '.nii.gz')):
            raise FileNotFoundError(f"Segmentation file not found for {file} in {seg_folder}")
    
    logger.info(f"Computing surface distances for {len(datalist)} cases")
    
    surface_distances_dict = {}
    
    for idx, file in enumerate(datalist):
        logger.info(f'Processing case: {file}, {idx+1}/{len(datalist)}')
        
        seg_file_path = os.path.join(seg_folder, file + '.nii.gz')
        gt_file_path = os.path.join(gt_folder, file + '.nii.gz')

        # Load files using MONAI transforms
        data = monai_transforms({'seg': seg_file_path, 'gt': gt_file_path})
        
        seg_tensor = data['seg'][0]  # First axis is channel dim
        gt_tensor = data['gt'][0]
        
        # Convert to numpy
        seg_array = seg_tensor.numpy()
        gt_array = gt_tensor.numpy()
        
        logger.info(f'  Seg shape: {seg_array.shape}, GT shape: {gt_array.shape}')
        
        # Compute surface distances
        surface_dists = compute_surface_distances(seg_array, gt_array)
        
        # DEBUG: Log surface distance summary
        logger.info(f'  Hausdorff distance: {surface_dists["hausdorff_distance"]:.4f}')
        logger.info(f'  GT->Seg mean distance: {surface_dists["dist_gt_to_seg_mean"]:.4f}')
        logger.info(f'  Seg->GT mean distance: {surface_dists["dist_seg_to_gt_mean"]:.4f}')
        
        # Store results (remove raw arrays from serializable dict)
        serializable_result = {k: v for k, v in surface_dists.items() if not k.startswith('_')}
        surface_distances_dict[file] = serializable_result
        
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
    parser.add_argument('--segmentation_base_folder', type=str, required=True, help='Path to the base folder containing dataset folders with segmentations.')
    parser.add_argument('--gt_base_folder', type=str, required=True, help='Path to the base folder containing dataset-separated folders with gold standard annotations.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset for reading segmentations and saving results.')
    parser.add_argument('--seg_subfolder_path', type=str, default='preds_final', help='Subfolder name for segmentations.')
    parser.add_argument('--gt_subfolder_path', type=str, default='labelsTs', help='Subfolder name for ground truth.')
    parser.add_argument('--output_results_base_folder', type=str, default=os.path.join(parent_dir, 'results_summary'), help='Path to base folder for saving results.')
    parser.add_argument('--reference_splits_base_folder', type=str, default=os.path.join(parent_dir, 'datasets'), help='Path to base folder containing dataset_split.json files.')
    parser.add_argument('--data_split', type=str, default='test', choices=['train', 'test'], help='Data split to use.')
    parser.add_argument('--strategy_type', type=str, default='all', choices=['all', 'kfold'], help='Sampling strategy type.')
    parser.add_argument('--total_folds', type=int, default=None, help='Total number of folds for kfold strategy.')
    args = parser.parse_args()

    output_folder = os.path.join(args.output_results_base_folder, args.dataset_name, 'nnUNet_surface_distances')
    logger = setup_logging(output_folder)
    
    logger.info(f"Starting surface distance calculation for dataset: {args.dataset_name}")
    logger.info(f"Segmentation folder: {os.path.join(args.segmentation_base_folder, args.dataset_name, args.seg_subfolder_path)}")
    logger.info(f"Ground truth folder: {os.path.join(args.gt_base_folder, args.dataset_name, args.gt_subfolder_path)}")
    
    seg_folder = os.path.join(args.segmentation_base_folder, args.dataset_name, args.seg_subfolder_path)
    gt_folder = os.path.join(args.gt_base_folder, args.dataset_name, args.gt_subfolder_path)

    # Read data split
    data_split_path = os.path.join(args.reference_splits_base_folder, args.dataset_name, 'dataset_split.json')
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
        datalist = []
        for path in dictionary_split_paths:
            datalist.extend(extractor(data_split, path))
    
    logger.info(f"Number of cases in datalist: {len(datalist)}")
    
    # Placeholder for config_labels_dict (not strictly needed for surface distance but keeping for consistency)
    config_labels_dict = {}
    
    # Calculate surface distances
    surface_distances_dict = calculate_surface_distances(
        seg_folder, 
        gt_folder, 
        config_labels_dict, 
        datalist,
        logger
    )
    
    # Save results as JSON
    output_json = os.path.join(output_folder, 'surface_distances.json')
    os.makedirs(output_folder, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(surface_distances_dict, f, indent=2)
    
    logger.info(f"Saved surface distance results to {output_json}")
    logger.info(f"Completed surface distance calculation for {len(surface_distances_dict)} cases")
