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
from skimage.measure import label as cc_label
from utils import (
    load_json, 
    save_json, 
    full_path_splitter, 
    file_ext_splitter,
)

def get_largest_cc(mask):
    #We are going to adapt this from our dataloading, and hardcode it to be for binary mask because lung is a binary seg task.

    largest_cc = np.zeros(shape=mask.shape, dtype=mask.dtype)
    if not largest_cc.ndim == 3:
        raise Exception('Currently we are only working in the domain of volumetric segmentation.')

    analysed_mask = mask 

    fg_mask = np.where(analysed_mask == 1, 1, 0).astype(dtype=np.int32) 
    #hard coded for volumetric tasks also.
    cc_analysed_mask, num_cc = cc_label(label_image=fg_mask, background=0, return_num=True, connectivity=3)
    if num_cc == 0 or cc_analysed_mask.sum() == 0:
        raise Exception('Lung doesnt have any empty targets') 
    else:
        largest_region_code = np.argmax(np.bincount(cc_analysed_mask.flat)[1:]) + 1 # + 1 because we indexed out the bg, 
        #but np.argmax will return values 0 indexed, which could give us largest_region_code = 0 despite this being bg!!
        if not largest_region_code > 0 or not largest_region_code <= num_cc:
            raise ValueError('Somehow we managed to get an invalid number of cc under the subloop which handles cases which were not empty')
        largest_cc = np.where(cc_analysed_mask == largest_region_code, 1, 0).astype(dtype=mask.dtype)
        # warnings.warn(f'Not really a warning, but voxel count in remaining component for semantic class {class_lb} had {stored_ccs[class_lb].sum()} voxels')
        if num_cc == 1: #largest_region_code == 1: 
            extracted_cc_bool = False
        else:
            extracted_cc_bool = True
    return largest_cc, extracted_cc_bool


def binarise_semantic_seg_nifti(
        case_name, 
        orig_folder, 
        output_folder,
        foreground_class_lb,
        extract_largest_cc=True
        # foreground_class_code=1, 
        ):
    '''
    Function intended for merging binary segs for each semantic class and writing them in nifti format:
    
    case_name = case_name
    orig_folder = path to the original directory for all cases. 
    output_folder = path to segmentation directory for the given case
    foreground_class_lb = the list of foreground classes to be merged.
    foreground_class_code = the code for the output foreground class
    
    We don't need to specify the background class because those classes would just get assigned to 0, so we will just pass over them.
    '''
    if len(foreground_class_lb) == 0:
        raise Exception("No foreground classes provided for binarisation.")
    # if len(background_class_lb) == 0:
    #     raise Exception("No background classes provided for binarisation.")
     
    #No need to read any background labels! Lets just merge the foreground classes. 
    fgs = None 
    spacing = None 
    origin = None 
    direction = None 
    for class_lb in foreground_class_lb: 
        img_itk = sitk.ReadImage(os.path.join(orig_folder, case_name, 'annotator_1', 'semantic_class_%s' % class_lb, f'{case_name}_0001.nii.gz'))
        dim = img_itk.GetDimension()

        if spacing is not None and spacing != img_itk.GetSpacing():
            raise RuntimeError("Inconsistent spacing in file %s, expected %s, got %s" % (case_name, spacing, img_itk.GetSpacing()))
        else:
            spacing = img_itk.GetSpacing()
        if origin is not None and origin != img_itk.GetOrigin():
            raise RuntimeError("Inconsistent origin in file %s, expected %s, got %s" % (case_name, origin, img_itk.GetOrigin()))
        else:
            origin = img_itk.GetOrigin()
        if direction is not None and not np.all(np.isclose(np.array(direction), np.array(img_itk.GetDirection()))):
            raise RuntimeError("Inconsistent direction in file %s, expected %s, got %s" % (case_name, direction, np.array(img_itk.GetDirection()).reshape(4,4)))
        else:
            direction = img_itk.GetDirection()

        if fgs is None:
            fgs = sitk.GetArrayFromImage(img_itk)
        else:
            fgs += sitk.GetArrayFromImage(img_itk)

    if np.any(fgs < 0) or np.any(fgs > 1):
        raise ValueError('Binary seg masks should only contain 0s or 1s')
    
    #If we need the largest connected component we can do that here. 
    if extract_largest_cc == True:
        fgs, extracted_cc_bool = get_largest_cc(fgs)
        if extracted_cc_bool:
            print(f'Extracted biggest component in case: {case_name} largest connected component voxel count: {fgs.sum()}')
    else:
        pass 

    #We presume only 3D volumes for the segmentations! 
    if dim != 3:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, case_name))
    else:
        seg_itk_new = sitk.GetImageFromArray(fgs.astype(np.uint8))  # Convert to uint8 for binary mask
        seg_itk_new.SetSpacing(spacing)
        seg_itk_new.SetOrigin(origin)
        seg_itk_new.SetDirection(direction)
        
        sitk.WriteImage(seg_itk_new, os.path.join(output_folder, f"{case_name}.nii.gz")) 
    
def convert_dataset(
        our_processed_folder: str,
        target_dataset_dir_path:str,
        task_config_path: str,   
        split_name: str,
        task_id: int,        
        dataset_name: str,
        # extract_largest_cc: bool = False,         
        num_processes: int = 1) -> None:
    if our_processed_folder.endswith('/') or our_processed_folder.endswith('\\'):
        our_processed_folder = our_processed_folder[:-1] 

    target_folder = os.path.join(target_dataset_dir_path, dataset_name)
    os.makedirs(target_folder, exist_ok=True)

    if split_name == 'train':
        labels_path = os.path.join(our_processed_folder, 'labelsTr')
        images_path = os.path.join(our_processed_folder, 'imagesTr')
        assert os.path.isdir(labels_path) and os.path.exists(labels_path), f"labelsTr missing in source folder or was not a subdirectory."
        assert os.path.isdir(images_path) and os.path.exists(images_path), f"imagesTr missing in source folder or was not a subdirectory."
        target_images = os.path.join(target_folder, 'imagesTr')
        target_labels = os.path.join(target_folder, 'labelsTr')
        os.makedirs(target_images, exist_ok=True)
        os.makedirs(target_labels, exist_ok=True)

    elif split_name == 'test':
        labels_path = os.path.join(our_processed_folder, 'labelsTs')
        images_path = os.path.join(our_processed_folder, 'imagesTs')

        assert os.path.isdir(images_path) and os.path.exists(images_path), f"imagesTs missing in source folder or was not a subdirectory."
        assert os.path.isdir(labels_path) and os.path.exists(labels_path), "labelsTs missing in source folder or was not a subdirectory."
        target_images = os.path.join(target_folder, 'imagesTs') 
        target_labels = os.path.join(target_folder, 'labelsTs')
        os.makedirs(target_images, exist_ok=True)
        os.makedirs(target_labels, exist_ok=True)
    else:
        raise ValueError(f"Unsupported split name {split_name}, only 'train' and 'test' are supported.")
    

    #All json related information will be extracted from our reformatted dataset. 
    
    #Extracting the task config file for merging. 

    #Loading the dataset.json from our reformatted dataset for reference on the image channels. 
    our_dataset_json = os.path.join(our_processed_folder, 'dataset.json')
    our_dataset_json = load_json(our_dataset_json)

    #Loading the exp task configs so that we can use it to only select the image channels we want.......
    task_configs = load_json(task_config_path)
    channel_codes_our_json = our_dataset_json['channel_names']
    channel_lbs_our_task = task_configs[f'task_id_{task_id}']['data_sampling']['image_conf']['image_channel']

    if len(channel_lbs_our_task) != 1:
        raise Exception("Current only single channel implementations are being evaluated, hence multiple channels will not yet be supported for consistency.")
    
    if channel_lbs_our_task[0] not in channel_codes_our_json:
        raise Exception(f"Channel {channel_lbs_our_task[0]} not found in our dataset json. Please check the task config file.")
    #Extracting the relevant channel code.
    channel_code = our_dataset_json['channel_names'][channel_lbs_our_task[0]]

    #Extracting the list of cases which will need to be processed.
    data_split = os.path.join(our_processed_folder, 'dataset_split.json')
    data_split = load_json(data_split)
    print(data_split.keys())
    if f'all_{split_name}' not in data_split['sampling'].keys():
        raise Exception(f"Expected all_{split_name} key in dataset split json for processing, but it was not found. Please check the dataset split json.")
    list_of_cases = data_split['sampling'][f'all_{split_name}']['all_cases']

    print(f"Checking for any data transforms specified in task config for dataset conversion...")
    #Setting a default value for extract_largest_cc, hacky fix.
    extract_largest_cc = False

    for key, transforms in task_configs[f'task_id_{task_id}']['data_transforms'].items():
        if 'semantic_class_mapping' == key:     
            class_labels_our_task = task_configs[f'task_id_{task_id}']['data_transforms']['semantic_class_mapping'] 
            #Extracting the list of foreground and background labels for mapping.
            output_semantic_class_dict = dict()
            for class_lb in class_labels_our_task.keys():
                if class_lb == 'background':
                    output_semantic_class_dict['background'] = class_labels_our_task[class_lb]
                    print(f'Background class labels correspond to original class labels of: {output_semantic_class_dict["background"]}.')
                else:
                    output_semantic_class_dict[class_lb] = class_labels_our_task[class_lb] 
                    print(f'{class_lb} labels correspond to original class labels of {output_semantic_class_dict[class_lb]} under key: {class_lb}.')

        elif 'component_extraction' == key:
            if transforms == 'cc_largest':
                extract_largest_cc = True
                print(f'Extracting largest cc? : {extract_largest_cc}')
            else:
                print(f'Extracting largest cc? : {extract_largest_cc}') 
        else:
            raise Exception('Unsupported data transform found in task config for conversion process.')
    
    if len(output_semantic_class_dict.keys()) == 2:
        foreground_key = [key for key in output_semantic_class_dict.keys() if key != 'background'][0]
        foreground_labels = output_semantic_class_dict[foreground_key]
        print(f'Foreground labels for binarisation: {foreground_labels} under foreground key: {foreground_key}')
        print(f'Background labels for binarisation: {output_semantic_class_dict["background"]}')
        print(f'Largest CC extraction set to: {extract_largest_cc}')
        for case in list_of_cases:
            shutil.copy(os.path.join(images_path, case, f'{case}' + '_%04.0d.nii.gz' % int(channel_code)), os.path.join(target_images, f'{case}' + '_%04.0d.nii.gz' % 0)) 
            #For now it is always single channel, so it always maps to that too!
            foreground_key = [key for key in output_semantic_class_dict.keys() if key != 'background'][0]
            foreground_labels = output_semantic_class_dict[foreground_key]
            binarise_semantic_seg_nifti(case_name=case, orig_folder=labels_path, output_folder=target_labels, foreground_class_lb=foreground_labels, extract_largest_cc=extract_largest_cc)
    else:
        raise Exception("Current only binarisation in this conversion script.")
    

    if os.path.exists(os.path.join(target_folder, 'dataset.json')):
        existing_json = load_json(os.path.join(target_folder, 'dataset.json'))
        if split_name == 'train':
            existing_json.update({
                'numTraining': len(list_of_cases)
            })
        else:
            existing_json.update({
                'numTest': len(list_of_cases)
            })
        output_json = existing_json
    else:
        if split_name == 'train':
            numTraining = len(list_of_cases)
            output_json = {
            'channel_names': {"0": channel_lbs_our_task[0]}, #single channel.
            'labels': {'background': 0, task_configs['task_id_1']['seg_problem']:1},
            'numTraining': numTraining,
            'file_ending': '.nii.gz',
            }
        else:
            numTest = len(list_of_cases)
            output_json = {
            'channel_names': {"0": channel_lbs_our_task[0]}, #single channel.
            'labels': {'background': 0, task_configs['task_id_1']['seg_problem']:1},
            'numTest': numTest,
            'file_ending': '.nii.gz',
            }

    save_json(output_json, os.path.join(target_folder, 'dataset.json'), sort_keys=False)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--reference_dataset_path', type=str, required=True, help='Path to the reference dataset which is to be used for processing.')
    argparser.add_argument('--target_dataset_basedir_path', type=str, required=True, help='Path to the target directory where the converted dataset will be stored.')
    argparser.add_argument('--task_config_basepath', type=str, required=True, help='Path to the task config file for the given dataset')
    argparser.add_argument('--split_name', type=str, default='train', help='Name of the split being processed')
    argparser.add_argument('--reference_task_id', type=int, default=1, help='Task ID in the reference dataset task config to be used for the conversion process, \n' \
    'it is typically assumed that the information from this task configuration is consistent across train and test splits for the processing applied here!')
    # argparser.add_argument('--extract_largest_cc', action='store_true', default=False, help='Whether to extract the largest connected component for relevant semantic classes (e.g., lung).')
    argparser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use for conversion.')
    args = argparser.parse_args()
    dataset_name = os.path.basename(os.path.dirname(args.reference_dataset_path))
    print(dataset_name)
    task_config_path = os.path.join(args.task_config_basepath, dataset_name, 'task_configs.txt')
    convert_dataset(
        args.reference_dataset_path, 
        args.target_dataset_basedir_path, 
        task_config_path, 
        args.split_name, 
        args.reference_task_id, 
        dataset_name, 
        # args.extract_largest_cc, 
        args.num_workers)