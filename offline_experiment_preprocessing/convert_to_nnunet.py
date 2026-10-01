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

codebase_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, codebase_dir)
from src.general_utils.dict_utils import extract_config, extractor, dict_deep_equals

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


def resolve_exp_configs(experiment_manifest_path: str, exp_config_ids: list[int]) -> dict:
    '''
    Loads experiment_manifest.json once and returns the requested exp_config_N entries.
    Unlike task_id, exp_config_id maps to exactly one task+prompter+metrics combination by
    construction, so this is a direct lookup, not a scan.
    '''
    manifest = extract_config(experiment_manifest_path, None)
    resolved = {}
    for eid in exp_config_ids:
        key = f'exp_config_{eid}'
        if key not in manifest:
            raise Exception(f"{key} not found in experiment manifest {experiment_manifest_path}.")
        resolved[key] = manifest[key]
    return resolved


def convert_dataset(
        our_processed_folder: str,
        target_dataset_dir_path:str,
        experiment_manifest_path: str,
        split_name: str,
        exp_config_ids: list[int],
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

    #Loading the relevant per-exp-config-id configs from the experiment manifest so that we can use it
    #to only select the image channels we want.......
    exp_configs = resolve_exp_configs(experiment_manifest_path, exp_config_ids)
    channel_codes_our_json = our_dataset_json['channel_names']
    if isinstance(exp_config_ids, list):
        channel_lbs_our_task = []
        seg_problem_our_task = []
        sample_group_categories_our_task = []
        for eid in exp_config_ids:
            task_cfg = exp_configs[f'exp_config_{eid}']['task']['config']
            channel_lbs_our_task.append(extractor(task_cfg, ('data_sampling', 'image_conf', 'image_channel')))
            seg_problem_our_task.append(extractor(task_cfg, ('seg_problem',)))
            sample_group_categories_our_task.append(extractor(task_cfg, ('data_sampling', 'sample_group_category')))
        #Now, we will assert that we have the same labels.
        for other_channel_lb in channel_lbs_our_task[1:]:
            eq, diffs = dict_deep_equals(channel_lbs_our_task[0], other_channel_lb)
            if not eq:
                raise Exception(f"Multiple exp config ids provided, but they do not have the same image channel labels. Please provide exp config ids with consistent image channel labels for the conversion process. Got channel labels: {channel_lbs_our_task}. Diffs: {diffs}")
        for other_seg_problem in seg_problem_our_task[1:]:
            eq, diffs = dict_deep_equals(seg_problem_our_task[0], other_seg_problem)
            if not eq:
                raise Exception(f"Multiple exp config ids provided, but they do not have the same seg_problem. Got seg_problems: {seg_problem_our_task}. Diffs: {diffs}")
        channel_lbs_our_task = channel_lbs_our_task[0] #We will just take the first one since we have asserted they are all the same.
        seg_problem_our_task = seg_problem_our_task[0] #We will just take the first one since we have asserted they are all the same.
    else:
        # channel_lbs_our_task = exp_configs[f'exp_config_{exp_config_id}']['task']['config']['data_sampling']['image_conf']['image_channel']
        raise Exception("The argument for exp_config_ids is now expected to be a list of exp config ids, but a single id was provided. Please provide a list of exp config ids with consistent image channel labels for the conversion process. Got a single id: %s" % exp_config_ids)

    if len(channel_lbs_our_task) != 1:
        raise Exception("Current only single channel implementations are being evaluated, hence multiple channels will not yet be supported for consistency.")

    if channel_lbs_our_task[0] not in channel_codes_our_json:
        raise Exception(f"Channel {channel_lbs_our_task[0]} not found in our dataset json. Please check the task config file.")
    #Extracting the relevant channel code.
    channel_code = our_dataset_json['channel_names'][channel_lbs_our_task[0]]

    #Extracting the list of cases which will need to be processed. The selection is driven by the
    #sample_group_category of each given exp_config_id -- a real filter (union of only the folds
    #actually named by the given ids), not an unconditional union of every fold in the file.
    data_split = os.path.join(our_processed_folder, 'dataset_split.json')
    data_split = load_json(data_split)
    print(data_split.keys())
    if f'all_{split_name}' in data_split['sampling'].keys():
        # raise Exception(f"Expected all_{split_name} key in dataset split json for processing, but it was not found. Please check the dataset split json.")
        list_of_cases = data_split['sampling'][f'all_{split_name}']['all_cases']
    else:
        list_of_cases = []
        seen_cases = set()
        for sample_group_category in sample_group_categories_our_task:
            category_type, category_value = sample_group_category
            if not (category_type.startswith('kfold') and category_type.endswith(f'_{split_name}')):
                raise Exception(f"Unsupported sample_group_category {sample_group_category} for split "
                                 f"{split_name}; expected a 'kfold*_{split_name}' category type since no "
                                 f"all_{split_name} key exists in the dataset split json.")
            if category_type not in data_split['sampling']:
                raise Exception(f"sample_group_category type {category_type} (from a reference exp config id's "
                                 f"config) not found in dataset_split.json['sampling'] keys: "
                                 f"{list(data_split['sampling'].keys())}.")
            fold_names = category_value if isinstance(category_value, list) else [category_value]
            for fold_name in fold_names:
                for case in data_split['sampling'][category_type][fold_name]:
                    if case not in seen_cases:
                        seen_cases.add(case)
                        list_of_cases.append(case)
        assert len(list_of_cases) > 0, (f"No cases resolved for split {split_name} from the given reference "
                                         f"exp config ids' sample_group_categories: {sample_group_categories_our_task}.")

    print(f"Checking for any data transforms specified in task config for dataset conversion...")

    prev_semantic_class_dict = None
    prev_extract_largest_cc = None

    if not isinstance(exp_config_ids, list):
        raise Exception("The argument for exp_config_ids is now expected to be a list of exp config ids, but a single id was provided. Please provide a list of exp config ids with consistent data transform settings for the conversion process. Got a single id: %s" % exp_config_ids)

    for eid in exp_config_ids:
        #We will just put assertions that it is fixed
        #config across all relevant folds!

        #Setting a default value for extract_largest_cc, hacky fix.
        extract_largest_cc = False
        task_transforms = exp_configs[f'exp_config_{eid}']['task']['config']['data_transforms']
        for key, transforms in task_transforms.items():
            if 'semantic_class_mapping' == key:
                class_labels_our_task = task_transforms['semantic_class_mapping']
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
                    eq, diffs = dict_deep_equals(output_semantic_class_dict, prev_semantic_class_dict)
                    assert eq, f"Semantic class mapping specified in task config for conversion process is not consistent across folds. Got {output_semantic_class_dict} and {prev_semantic_class_dict} in different folds. Diffs: {diffs}"
                else:
                    prev_semantic_class_dict = copy.deepcopy(output_semantic_class_dict)

            elif 'component_extraction' == key:
                if transforms == 'cc_largest':
                    extract_largest_cc = True
                    print(f'Extracting largest cc? : {extract_largest_cc}')
                else:
                    print(f'Extracting largest cc? : {extract_largest_cc}')

                if prev_extract_largest_cc is not None:
                    eq, diffs = dict_deep_equals(extract_largest_cc, prev_extract_largest_cc)
                    assert eq, f"Component extraction setting specified in task config for conversion process is not consistent across folds. Got {extract_largest_cc} and {prev_extract_largest_cc} in different folds. Diffs: {diffs}"
                else:
                    prev_extract_largest_cc = copy.deepcopy(extract_largest_cc)
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
    argparser.add_argument('--task_config_basepath', type=str, required=True, help='Path to the exp_configs base directory for the given dataset (used to locate experiment_manifest.json)')
    argparser.add_argument('--split_name', type=str, default='train', help='Name of the split being processed')
    argparser.add_argument('--reference_exp_config_ids', type=int, nargs='+', default=[1], help='exp_config IDs (from experiment_manifest.json) to be used for the conversion process, \n' \
    'it is typically assumed that the information from these exp configs is consistent across train and test splits for the processing applied here!')
    # argparser.add_argument('--extract_largest_cc', action='store_true', default=False, help='Whether to extract the largest connected component for relevant semantic classes (e.g., lung).')
    argparser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use for conversion.')
    args = argparser.parse_args()
    dataset_name = os.path.basename(os.path.dirname(args.reference_dataset_path))
    print(dataset_name)
    experiment_manifest_path = os.path.join(args.task_config_basepath, dataset_name, 'experiment_manifest.json')
    convert_dataset(
        args.reference_dataset_path,
        args.target_dataset_basedir_path,
        experiment_manifest_path,
        args.split_name,
        args.reference_exp_config_ids,
        dataset_name,
        # args.extract_largest_cc,
        args.num_workers)
