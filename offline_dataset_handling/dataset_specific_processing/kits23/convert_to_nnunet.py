#This script is intended for converting semantic segmentation datasets from the format provided to that required by nnU-Net as this serves as one of the baseline
#comparisons for determining segmentation convergence. Only supports implementation for 

import argparse
import multiprocessing
import shutil
from typing import Optional
import SimpleITK as sitk
import itk
import os 
import sys 
import numpy as np
import json
import warnings
import torch
import copy
import re
from skimage.measure import label as cc_label
offline_dhandling_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(offline_dhandling_dir)
sys.path.insert(0, offline_dhandling_dir)
from utils import (
    load_json, 
    save_json, 
    full_path_splitter, 
    file_ext_splitter,
)

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CropForegroundd,
)
#We for now will assume our transforms are fixed configuration so that we don't need dynamic initialisation
#according to non fixed parameters. 
transforms_dict = {
    'foreground_crop': CropForegroundd(
        keys=['image', 'seg'],
        source_key='seg',
        margin=30, #We will set a fixed margin of 30 voxels. 
        allow_smaller=True),
}
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
        annotator_id=1,
        instance_id=1,
        extract_largest_cc=True
        ):
    '''
    Function intended for merging binary segs for each semantic class
    
    case_name = case_name
    orig_folder = path to the original directory for all cases. 
    output_folder = path to segmentation directory for the given case
    foreground_class_lb = the list of foreground classes to be merged.
    annotator_id = the annotator id for which the segmentations will be read and merged.
    instance_id = the instance id for which the segmentations will be read and merged.
    extract_largest_cc = whether to extract the largest connected component for the merged binary segmentation mask, default is True.

    '''
    if len(foreground_class_lb) == 0:
        raise Exception("No foreground classes provided for binarisation.")
    
    #No need to read any background labels! Lets just merge the foreground classes. 
    fgs = None 
    spacing = None 
    origin = None 
    direction = None 
    for class_lb in foreground_class_lb: 
        img_itk = sitk.ReadImage(os.path.join(orig_folder, case_name, f'annotator_{annotator_id}', 'semantic_class_%s' % class_lb, f'{case_name}_{"{:04d}".format(instance_id)}.nii.gz'))
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

    #Now we will perform any image augmentations if specified in the task config. 

    if dim != 3:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, case_name))
    else:
        seg_itk_new = sitk.GetImageFromArray(fgs.astype(np.uint8))  # Convert to uint8 for binary mask
        seg_itk_new.SetSpacing(spacing)
        seg_itk_new.SetOrigin(origin)
        seg_itk_new.SetDirection(direction)
        
        sitk.WriteImage(seg_itk_new, os.path.join(output_folder, f"{case_name}.nii.gz")) 
        # return seg_itk_new
    
def apply_image_augs(
    case_name: str,
    image_folder: str,
    seg_folder: str,
    augmentations: list[str]
):
    #This function will apply any specified image augmentations and write the final image and segmentations.
    
    monai_transforms = [
        LoadImaged(keys=['image', 'seg'], reader='ITKReader', image_only=True),
        EnsureChannelFirstd(keys=['image', 'seg']),
    ]

    if augmentations is not None:
        print(f"Applying the following augmentations (in order) specified in task config for conversion process: {augmentations}")
        monai_transforms.extend([
            transforms_dict[aug]        for aug in augmentations 
        ])

    input_dict = {
        'image': os.path.join(image_folder, f'{case_name}_0000.nii.gz'),
        'seg': os.path.join(seg_folder, f'{case_name}.nii.gz')
    }
    processed_dict = Compose(monai_transforms)(input_dict)
    

    #now we will write the processed image and seg to the target folders
     

    image_np = processed_dict['image']
    seg_np = processed_dict['seg']
    image_np = image_np[0]
    seg_np = seg_np[0]

    image_affine = processed_dict['image'].meta['affine'] 
    seg_affine = processed_dict['seg'].meta['affine']
    assert torch.allclose(image_affine, seg_affine), f"Image and segmentation affines are not close for case {case_name}, got {image_affine} and {seg_affine}. Please check the original image and segmentation affines for this case."
    assert image_np.shape == seg_np.shape, f"Image and segmentation shapes are not the same for case {case_name}, got {image_np.shape} and {seg_np.shape}. Please check the original image and segmentation shapes for this case."
    print(image_np.meta)
    print(image_np.shape)
    print(seg_np.shape)
    if isinstance(image_np, torch.Tensor):
        image_np = image_np.numpy()
        seg_np = seg_np.numpy()
    if isinstance(image_affine, torch.Tensor):
        image_affine = image_affine.numpy()
    if len(image_np.shape) >= 2:
        image_np = image_np.transpose().copy() #Transposition operation is necessary to convert between the axis ordering in MONAI IO and ITK
        seg_np = seg_np.transpose().copy() #Transposition operation is necessary to convert between the axis ordering in MONAI IO and ITK

    image_np = image_np.astype(np.float32)
    seg_np = seg_np.astype(np.uint8)

    result_image = itk.image_from_array(image_np)
    result_seg = itk.image_from_array(seg_np)
    if image_affine is not None:
        
        convert_aff_mat = np.diag([-1, -1, 1, 1])
        if image_affine.shape[0] == 3:
            raise NotImplementedError('We do not yet provide handling for 2D images')
            # if affine.shape[0] == 3:  # Handle RGB (2D Image)
                # convert_aff_mat = np.diag([-1, -1, 1])

        image_affine = convert_aff_mat @ image_affine

        dim = image_affine.shape[0] - 1
        _origin_key = (slice(-1), -1)
        _m_key = (slice(-1), slice(-1))
        print(_m_key)
        origin = image_affine[_origin_key]
        spacing = np.linalg.norm(image_affine[_m_key] @ np.eye(dim), axis=0)
        print(spacing)
        direction = image_affine[_m_key] @ np.diag(1 / spacing)


        result_image.SetDirection(itk.matrix_from_array(direction))
        result_image.SetSpacing(spacing)
        result_image.SetOrigin(origin)
        result_seg.SetDirection(itk.matrix_from_array(direction))
        result_seg.SetSpacing(spacing)
        result_seg.SetOrigin(origin)

        itk.imwrite(result_image, os.path.join(image_folder, f'{case_name}_0000.nii.gz'), compression=True)
        itk.imwrite(result_seg, os.path.join(seg_folder, f'{case_name}.nii.gz'), compression=True)

    
def convert_dataset(
    our_processed_folder: str,
    target_dataset_dir_path:str,
    task_config_path: str,   
    split_name: str,
    task_ids: list[int],        
    output_name: str,
    # extract_largest_cc: bool = False,         
    num_processes: int = 1) -> None:
    if our_processed_folder.endswith('/') or our_processed_folder.endswith('\\'):
        our_processed_folder = our_processed_folder[:-1] 

    target_folder = os.path.join(target_dataset_dir_path, output_name)
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
    if isinstance(task_ids, list):
        channel_lbs_our_task = []
        seg_problem_our_task = []
        for tid in task_ids:
            channel_lbs_our_task.append(task_configs[f'task_id_{tid}']['data_sampling']['image_conf']['image_channel'])
            seg_problem_our_task.append(task_configs[f'task_id_{tid}']['seg_problem'])
        #Now, we will assert that we have the same labels.
        if not all(channel_lb == channel_lbs_our_task[0] for channel_lb in channel_lbs_our_task) or not all(seg_problem == seg_problem_our_task[0] for seg_problem in seg_problem_our_task):
            raise Exception(f"Multiple task ids provided, but they do not have the same image channel labels. Please provide task ids with consistent image channel labels for the conversion process. Got channel labels: {channel_lbs_our_task}")
        else:
            channel_lbs_our_task = channel_lbs_our_task[0] #We will just take the first one since we have asserted they are all the same.
            seg_problem_our_task = seg_problem_our_task[0] #We will just take the first one since we have asserted they are all the same.   
    else:
        # channel_lbs_our_task = task_configs[f'task_id_{task_id}']['data_sampling']['image_conf']['image_channel']
        raise Exception("The argument for task_ids is now expected to be a list of task ids, but a single task id was provided. Please provide a list of task ids with consistent image channel labels for the conversion process. Got a single task id: %s" % task_ids)

    if len(channel_lbs_our_task) != 1:
        raise Exception("Current only single channel implementations are being evaluated, hence multiple channels will not yet be supported for consistency.")
    
    if channel_lbs_our_task[0] not in channel_codes_our_json:
        raise Exception(f"Channel {channel_lbs_our_task[0]} not found in our dataset json. Please check the task config file.")
    #Extracting the relevant channel code.
    channel_code = our_dataset_json['channel_names'][channel_lbs_our_task[0]]

    #Extracting the list of cases which will need to be processed.
    data_split = os.path.join(our_processed_folder, 'dataset_split.json')
    data_split = load_json(data_split)
    # print(data_split.keys())
    if f'all_{split_name}' in data_split['sampling'].keys():
        # raise Exception(f"Expected all_{split_name} key in dataset split json for processing, but it was not found. Please check the dataset split json.")
        list_of_cases = data_split['sampling'][f'all_{split_name}']['all_cases']
    else:
        kfold_key = [key for key in data_split['sampling'].keys() if key.startswith(f'kfold') and key.endswith(f'_{split_name}')]
        #lets take the first one we find.
        assert len(kfold_key) > 0, f"Expected kfold key for split {split_name} in dataset split json for processing, but it was not found. Please check the dataset split json."
        split_dict_name = kfold_key[0]
        list_of_cases = [] 
        for k, v in data_split['sampling'][split_dict_name].items():
            if k.startswith('fold'):
                list_of_cases.extend(v)
        
    print(f"Checking for any data transforms specified in task config for dataset conversion...")

    prev_semantic_class_dict = None
    prev_extract_largest_cc = None
    prev_image_augmentations = None

    if not isinstance(task_ids, list):
        raise Exception("The argument for task_ids is now expected to be a list of task ids, but a single task id was provided. Please provide a list of task ids with consistent data transform settings for the conversion process. Got a single task id: %s" % task_ids)
    
    for id in task_ids:
        #We will just put assertions that it is fixed
        #config across all relevant folds! 

        #Setting a default value for extract_largest_cc, hacky fix.
        extract_largest_cc = False
        for key, transforms in task_configs[f'task_id_{id}']['data_transforms'].items():
            if 'semantic_class_mapping' == key:     
                class_labels_our_task = task_configs[f'task_id_{id}']['data_transforms']['semantic_class_mapping'] 
                #Extracting the list of foreground and background labels for mapping.
                output_semantic_class_dict = dict()
                for class_lb in class_labels_our_task.keys():
                    if class_lb == 'background':
                        output_semantic_class_dict['background'] = class_labels_our_task[class_lb]
                        print(f'Background class labels correspond to original class labels of: {output_semantic_class_dict["background"]}.')
                    else:
                        output_semantic_class_dict[class_lb] = class_labels_our_task[class_lb] 
                        print(f'{class_lb} labels correspond to original class labels of {output_semantic_class_dict[class_lb]} under key: {class_lb}.')
                
                #here we set a copy to compare across folds.
                if prev_semantic_class_dict is not None:
                    assert output_semantic_class_dict == prev_semantic_class_dict, f"Semantic class mapping specified in task config for conversion process is not consistent across folds. Please ensure the semantic class mapping is consistent across folds for the conversion process. Got {output_semantic_class_dict} and {prev_semantic_class_dict} in different folds."
                else:
                    prev_semantic_class_dict = copy.deepcopy(output_semantic_class_dict)

            elif 'component_extraction' == key:
                if transforms == 'cc_largest':
                    extract_largest_cc = True
                    print(f'Extracting largest cc? : {extract_largest_cc}')
                else:
                    print(f'Extracting largest cc? : {extract_largest_cc}')
                
                if prev_extract_largest_cc is not None:
                    assert extract_largest_cc == prev_extract_largest_cc, f"Component extraction setting specified in task config for conversion process is not consistent across folds. Please ensure the component extraction setting is consistent across folds for the conversion process. Got {extract_largest_cc} and {prev_extract_largest_cc} in different folds."
                else:
                    prev_extract_largest_cc = copy.deepcopy(extract_largest_cc)
            elif 'image_augmentations' == key:
                print(f"Set of transforms specified (ordered) {transforms} in task config for conversion process under image_augmentations conversion.")
                if prev_image_augmentations is not None:
                    assert transforms == prev_image_augmentations, f"Image augmentation setting specified in task config for conversion process is not consistent across folds. Please ensure the image augmentation setting is consistent across folds for the conversion process. Got {transforms} and {prev_image_augmentations} in different folds." 
                else:
                    prev_image_augmentations = copy.deepcopy(transforms)
            else:
                raise Exception('Unsupported data transform found in task config for conversion process.')
    
    if len(output_semantic_class_dict.keys()) == 2:
        foreground_key = [key for key in output_semantic_class_dict.keys() if key != 'background'][0]
        foreground_labels = output_semantic_class_dict[foreground_key]
        print(f'Foreground labels for binarisation: {foreground_labels} under foreground key: {foreground_key}')
        print(f'Background labels for binarisation: {output_semantic_class_dict["background"]}')
        print(f'Largest CC extraction set to: {extract_largest_cc}')
        print(f'Image augmentations: {prev_image_augmentations}')
        
        for task_id in task_ids:
            for case in list_of_cases:
                # print(list_of_cases)
                shutil.copy(os.path.join(images_path, case, f'{case}' + '_%04.0d.nii.gz' % int(channel_code)), os.path.join(target_images, f'{case}' + '_%04.0d.nii.gz' % 0)) 
                #For now it is always single channel, so it always maps to that too!
                foreground_key = [key for key in output_semantic_class_dict.keys() if key != 'background'][0]
                foreground_labels = output_semantic_class_dict[foreground_key]

                annotator_ids = task_configs[f'task_id_{id}']['data_sampling']['annotation_conf']['annotator']
                assert len(annotator_ids) == 1, f"Multiple annotator ids specified for task id {id} in task config for conversion process, but currently only single annotator id is supported for conversion. Please provide a single annotator id for the conversion process. Got annotator ids: {annotator_ids} for task id: {id}"
                annotator_id = annotator_ids[0]
                annotator_id = re.search(r'\d+$', annotator_id).group()
                instance_ids = task_configs[f'task_id_{id}']['data_sampling']['annotation_conf']['instance_id']
                assert len(instance_ids) == 1, f"Multiple instance ids specified for task id {id} in task config for conversion process, but currently only single instance id is supported for conversion. Please provide a single instance id for the conversion process. Got instance ids: {instance_ids} for task id: {id}"
                instance_id = re.search(r'\d+$', instance_ids[0]).group()
                
                assert instance_id.isdigit(), f"Extracted instance id {instance_id} is not a digit, please check the format of the instance ids specified in the task config for conversion process. Got instance id: {instance_id}"
                assert annotator_id.isdigit(), f"Extracted annotator id {annotator_id} is not a digit, please check the format of the annotator ids specified in the task config for conversion process. Got annotator id: {annotator_id}"
                annotator_id = int(annotator_id)
                instance_id = int(instance_id)

                binarise_semantic_seg_nifti(
                    case_name=case, 
                    orig_folder=labels_path, 
                    output_folder=target_labels, 
                    foreground_class_lb=foreground_labels, 
                    annotator_id=annotator_id, 
                    instance_id=instance_id, 
                    extract_largest_cc=extract_largest_cc
                )
                apply_image_augs(
                    case_name=case,
                    image_folder=target_images,
                    seg_folder=target_labels,
                    augmentations=prev_image_augmentations,
                )
    
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
            'labels': {'background': 0, seg_problem_our_task:1},
            'numTraining': numTraining,
            'file_ending': '.nii.gz',
            }
        else:
            numTest = len(list_of_cases)
            output_json = {
            'channel_names': {"0": channel_lbs_our_task[0]}, #single channel.
            'labels': {'background': 0, seg_problem_our_task:1},
            'numTest': numTest,
            'file_ending': '.nii.gz',
            }

    save_json(output_json, os.path.join(target_folder, 'dataset.json'), sort_keys=False)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--reference_dataset_path', type=str, required=True, help='Path to the reference dataset which is to be used for processing.')
    argparser.add_argument('--target_dataset_basedir_path', type=str, required=True, help='Path to the target directory where the converted dataset will be stored.')
    argparser.add_argument('--output_name', type=str, required=True, help='Name of the target dataset to be used for the conversion process, e.g., TaskXXX_DatasetName')
    argparser.add_argument('--task_config_basepath', type=str, required=True, help='Path to the task config file for the given dataset')
    argparser.add_argument('--split_name', type=str, default='train', help='Name of the split being processed')
    argparser.add_argument('--reference_task_ids', type=int, nargs='+', default=[1], help='Task IDs in the reference dataset task config to be used for the conversion process, \n' \
    'it is typically assumed that the information from this task configuration is consistent across train and test splits for the processing applied here!')
    # argparser.add_argument('--extract_largest_cc', action='store_true', default=False, help='Whether to extract the largest connected component for relevant semantic classes (e.g., lung).')
    argparser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use for conversion.')
    args = argparser.parse_args()
    dataset_name = os.path.basename(os.path.dirname(args.reference_dataset_path))
    # print(dataset_name)
    task_config_path = os.path.join(args.task_config_basepath, dataset_name, 'task_configs.txt')
    convert_dataset(
        args.reference_dataset_path, 
        args.target_dataset_basedir_path, 
        task_config_path, 
        args.split_name, 
        args.reference_task_ids, 
        args.output_name,
        # args.extract_largest_cc, 
        args.num_workers)