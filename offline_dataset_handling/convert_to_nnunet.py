# 

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
        ori_dataset_path: str,
        target_dataset_dir_path:str,          
        dataset_name: str,            
        num_processes: int = 1) -> None:
    if our_processed_folder.endswith('/') or our_processed_folder.endswith('\\'):
        our_processed_folder = our_processed_folder[:-1] 


    labelsTr_path = os.path.join(our_processed_folder, 'labelsTr')
    imagesTr_path = os.path.join(our_processed_folder, 'imagesTr')
    # labelsTs_path = os.path.join(our_processed_folder, 'labelsTs')
    # imagesTs_path = os.path.join(our_processed_folder, 'imagesTs')
    
    assert os.path.isdir(labelsTr_path) and os.path.exists(labelsTr_path), f"labelsTr missing in source folder or was not a subdirectory."
    # assert os.path.isdir(imagesTs_path) and os.path.exists(imagesTs_path), f"imagesTs missing in source folder or was not a subdirectory."
    assert os.path.isdir(imagesTr_path) and os.path.exists(imagesTr_path), f"imagesTr missing in source folder or was not a subdirectory."
    # if process_labelsTs:
    #     assert os.path.isdir(labelsTs_path) and os.path.exists(labelsTs_path), "labelsTs missing in source folder or was not a subdirectory."
    
    target_folder = os.path.join(target_dataset_dir_path, dataset_name)
    target_imagesTr = os.path.join(target_folder, 'imagesTr')
    # target_imagesTs = os.path.join(target_folder, 'imagesTs')
    target_labelsTr = os.path.join(target_folder, 'labelsTr')
    # target_labelsTs = os.path.join(target_folder, 'labelsTs')

    os.makedirs(os.path.join(target_dataset_dir_path, dataset_name), exist_ok=True)

    os.makedirs(target_imagesTr, exist_ok=True)
    # os.makedirs(target_imagesTs, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)
    # if process_labelsTs:
    #     os.makedirs(target_labelsTs, exist_ok=True) 


    #Loading the original dataset json to be reformatted.
    orig_dataset_json = os.path.join(ori_dataset_path, 'dataset.json')
    assert os.path.isfile(orig_dataset_json), f"MSD formatted dataset.json was missing in source_folder"

    orig_dataset_json = load_json(orig_dataset_json)

    #Extracting the task config file for merging. 
    task_config_txt = os.path.join(f'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/exp_configs/{dataset_name}/task_configs.txt')

    #Loading the dataset.json from our reformatted dataset for reference on the image channels. 
    our_dataset_json = os.path.join(our_processed_folder, 'dataset.json')
    our_dataset_json = load_json(our_dataset_json)

    #Loading the exp task configs so that we can use it to only select the image channels we want.......
    task_configs = load_json(task_config_txt)
    channel_codes_our_json = our_dataset_json['channel_names']
    channel_lbs_our_task = task_configs['task_id_1']['data_sampling']['image_conf']['image_channel']

    class_labels_our_task = task_configs['task_id_1']['data_transforms']['semantic_class_mapping'] 
    for class_lb in class_labels_our_task.keys(): 
        if class_lb == 'background':
            background_labels = class_labels_our_task[class_lb]
            # background_code = 0
        else:
            foreground_labels = class_labels_our_task[class_lb]
            # foreground_code = 1 
    
    if len(channel_lbs_our_task) != 1:
        raise Exception 
    
    if channel_lbs_our_task[0] not in channel_codes_our_json:
        raise Exception(f"Channel {channel_lbs_our_task[0]} not found in our dataset json. Please check the task config file.")

    channel_code = our_dataset_json['channel_names'][channel_lbs_our_task[0]]

    list_of_cases = os.listdir(imagesTr_path)

    
    for case in list_of_cases:
        shutil.copy(os.path.join(imagesTr_path, case, f'{case}' + '_%04.0d.nii.gz' % int(channel_code)), os.path.join(target_imagesTr, f'{case}' + '_%04.0d.nii.gz' % 0)) #always single channel so always maps to that too!
        
        if dataset_name != 'Dataset006_Lung':
            binarise_semantic_seg_nifti(case_name=case, orig_folder=labelsTr_path, output_folder=target_labelsTr, foreground_class_lb=foreground_labels, extract_largest_cc=False)
        else:
            binarise_semantic_seg_nifti(case_name=case, orig_folder=labelsTr_path, output_folder=target_labelsTr, foreground_class_lb=foreground_labels, extract_largest_cc=True)

        #The disparity arises because we extract the biggest connected component for the lung lesion dataset during IS evaluation, so we need
        # to remain consistent with that.     
    
    #Now for the labels. We know that nnunet has region based training, but we want to just do the binary semantic segmentation, so we
    #will pre-binarise the labels.. 

    # orig_dataset_json['labels'] = {'background': 0,
    #                                 task_configs['task_id_1']['seg_problem']:1}
    
    # orig_dataset_json['file_ending'] = ".nii.gz"
    # orig_dataset_json["channel_names"] = 
    
    output_json = {
        'channel_names': {"0": channel_lbs_our_task[0]}, #single channel.
        'labels': {'background': 0, task_configs['task_id_1']['seg_problem']:1},
        'numTraining': orig_dataset_json['numTraining'], #Only have access to training set, so we will just use that.
        'file_ending': '.nii.gz',

    }

    save_json(output_json, os.path.join(target_folder, 'dataset.json'), sort_keys=False)
    

if __name__ == '__main__':
    dataset_name = 'Dataset008_HepaticVessel'
    msd_dataset_name = 'Task08_HepaticVessel' 
    input_data_path = f'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/{dataset_name}'
    orig_data_path = f'/home/parhomesmaeili/Radiology_Datasets/MSD/{msd_dataset_name}'
    target_path = '/home/parhomesmaeili/Helmholtz Group/MICCAI2025_nnunet'
    convert_dataset(input_data_path, orig_data_path, target_path, dataset_name, False)