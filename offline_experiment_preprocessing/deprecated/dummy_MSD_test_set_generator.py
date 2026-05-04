#Script for creating a dummy dataset with the labelsTs included in the folder in order to test the conversion script for the MSD formatted datasets.
import argparse
import multiprocessing
import shutil
from typing import Optional
import SimpleITK as sitk
import os 
import sys 
import numpy as np
import json
import copy 
datasets_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),'datasets')
from IS_Validate.dataset_conversion.utils import check_dataset_existence, extract_nifti_files, extract_subfiles, extract_subdirs, load_json, save_json, full_path_splitter, file_ext_splitter

def create_dummy_seg_4d_nifti(filename, output_folder):
    #We create a dummy test set which I do not have access to, in order to validate that the conversion scripts will work.

    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = os.path.basename(filename)
    if dim == 3:
        shutil.copy(filename, os.path.join(output_folder, file_base[:-7] + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, filename))
    else:
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection()).reshape(4,4)
        # now modify these to remove the fourth dimension
        spacing = tuple(list(spacing[:-1]))
        origin = tuple(list(origin[:-1]))
        direction = tuple(direction[:-1, :-1].reshape(-1))
        
        img = np.where(img_npy[3] > 3, 0, img_npy[3])
        img_itk_new = sitk.GetImageFromArray(img)
        img_itk_new.SetSpacing(spacing)
        img_itk_new.SetOrigin(origin)
        img_itk_new.SetDirection(direction)
        sitk.WriteImage(img_itk_new, os.path.join(output_folder, file_base + ".nii.gz"))

def create_dummy_MSD_ts_dataset(source_folder: str,
                        num_processes: int = 1) -> None:
    if source_folder.endswith('/') or source_folder.endswith('\\'):
        source_folder = source_folder[:-1] 

    labelsTr_path = os.path.join(source_folder, 'labelsTr')
    imagesTr_path = os.path.join(source_folder, 'imagesTr')
    labelsTs_path = os.path.join(source_folder, 'labelsTs')
    imagesTs_path = os.path.join(source_folder, 'imagesTs')
    
    assert os.path.isdir(labelsTr_path) and os.path.exists(labelsTr_path), f"labelsTr missing in source folder or was not a subdirectory."
    assert os.path.isdir(imagesTs_path) and os.path.exists(imagesTs_path), f"imagesTs missing in source folder or was not a subdirectory."
    assert os.path.isdir(imagesTr_path) and os.path.exists(imagesTr_path), f"imagesTr missing in source folder or was not a subdirectory."
    assert not os.path.isdir(labelsTs_path) and not os.path.exists(labelsTs_path), "labelsTs was not missing in source folder or was a subdirectory, we are trying to create a dummy test."
    

    dataset_json = os.path.join(source_folder, 'dataset.json')
    assert os.path.isfile(dataset_json), f"MSD formatted dataset.json was missing in source_folder"

    target_folder = source_folder + 'dummy'
        

    target_imagesTr = os.path.join(target_folder, 'imagesTr')
    target_imagesTs = os.path.join(target_folder, 'imagesTs')
    target_labelsTr = os.path.join(target_folder, 'labelsTr')
    target_labelsTs = os.path.join(target_folder, 'labelsTs')

    shutil.copytree(imagesTr_path, target_imagesTr)
    shutil.copytree(labelsTr_path, target_labelsTr)
    shutil.copytree(imagesTs_path, target_imagesTs)
    os.makedirs(target_labelsTs)

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []

        # convert 4d test images to dummy segmentations
        source_images = [i for i in extract_nifti_files(imagesTs_path, join=False, sort=True) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [os.path.join(imagesTs_path, i) for i in source_images]
        target_sample_folders = [target_labelsTs] * len(source_images)
        results.append(
            p.starmap_async(
                create_dummy_seg_4d_nifti, zip(source_images, target_sample_folders)
            )
        )

        [i.get() for i in results]

    original_json = load_json(os.path.join(source_folder, 'dataset.json'))
    dummy_json = copy.deepcopy(original_json)
    dummy_test_relpaths = [] 
    for case in original_json['test']:
        case_filename = full_path_splitter(case)[-1] #MSD convention is to use the same filename, but just under different folders
        case_subdict = {
            'image': case, 
            'label': f'./labelsTs/' + case_filename
        }
        dummy_test_relpaths.append(case_subdict)
    del dummy_json['test']
    dummy_json['test'] = dummy_test_relpaths    
    save_json(dummy_json, os.path.join(target_folder, 'dataset.json'), indent=1, sort_keys=False)

def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_abs_path', type=str, required=True,
                        help='Downloaded and extracted MSD dataset folder absolute path. Example: '
                             '/home/parhomesmaeili/Radiology_Datasets/Task01_BrainTumour')
    parser.add_argument('-np', type=int, required=False, default=1,
                        help=f'Number of processes used. Default: 1')
    args = parser.parse_args()
    create_dummy_MSD_ts_dataset(args.i, args.np)


if __name__ == '__main__':
    import random
    create_dummy_MSD_ts_dataset('/home/parhomesmaeili/Radiology_Datasets/MSD/Task01_BrainTumour', num_processes=4)
