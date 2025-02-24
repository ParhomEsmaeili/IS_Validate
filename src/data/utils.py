from monai.transforms import Compose, LoadImaged, Orientationd, EnsureChannelFirstd, EnsureTyped
from monai.data import Dataset, MetaTensor 
import json 
import torch  
import numpy as np
from typing import Union, Optional 
import os
import copy 
import logging
import warnings 

# logger = logging.getLogger(__name__)

def dataset_generator(data_dict):
    '''
    This function handles the construction of a dataset object for iterating through.
    '''
    load_transforms = [
        LoadImaged(keys=['image', 'label'], reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        EnsureTyped(keys=['image', 'label'], dtype=[torch.float32, torch.int32]),
    ]
    return Dataset(data_dict, load_transforms)

def data_instance_generator(data_instance:dict):
     
    '''
    This function reformats the data instance from the output of the dataset generator's load transforms into the 
    data_instance reformat expected by the pseudo-ui.
    '''
    if not isinstance(data_instance, dict) or not data_instance:
        raise Exception('Data instance is assumed to be a non-empty dictionary format..')

    #We extract the paths from the meta_dict:

    im_path = copy.deepcopy(data_instance['image_meta_dict']['filename_or_obj'])
    label_path = copy.deepcopy(data_instance['label_meta_dict']['filename_or_obj'])

    if not os.path.exists(im_path) or not os.path.exists(label_path):
        raise Exception('One of the paths does not exist! Or the full absolute path was not provided to the dataset constructor')
    
    im_tensor = copy.deepcopy(data_instance['image'])
    label_tensor = copy.deepcopy(data_instance['label'])

    if not isinstance(im_tensor, torch.Tensor) or not isinstance(im_tensor, MetaTensor):
        raise TypeError('Image tensor was not a torch tensor or a Monai meta-tensor.')
    if not isinstance(label_tensor, torch.Tensor) or not isinstance(label_tensor, MetaTensor):
        raise TypeError('Label tensor was not a torch tensor or a Monai meta-tensor.')
    
    im_meta_dict = copy.deepcopy(data_instance['image_meta_dict'])
    label_meta_dict = copy.deepcopy(data_instance['image_meta_dict'])

    if not isinstance(im_meta_dict, dict) or not im_meta_dict:
        raise Exception('The image meta_dict must be a non-empty dictionary')
    
    if not isinstance(label_meta_dict, dict) or not label_meta_dict:
        raise Exception('The label meta_dict must be a non-empty dictionary') 

    #Checking that the meta dict contains the affine arrays required.

    original_affine_key = 'original_affine'
    current_affine_key = 'affine'

    def affine_to_tensor(meta_dict:dict, affine_key: str):
        if isinstance(meta_dict[affine_key], np.ndarray):
            warnings.warn(f'The meta_dicts key: {affine_key} was a numpy array, converting to torch tensor.')
            meta_dict[affine_key] = torch.from_numpy(meta_dict[affine_key])
            return meta_dict 
        elif isinstance(meta_dict[affine_key], torch.Tensor):
            return meta_dict
        else:
            raise TypeError('The affine info must be initially presented as a numpy or torch tensor.')
    #Running the affine checker for the original and current affine; for both the image and the label
    im_meta_dict = affine_to_tensor(im_meta_dict, original_affine_key)
    im_meta_dict = affine_to_tensor(im_meta_dict, current_affine_key)

    label_meta_dict = affine_to_tensor(label_meta_dict, original_affine_key)
    label_meta_dict = affine_to_tensor(label_meta_dict, current_affine_key)
    

    #Now populating the reformatted data instance:

    reformat_data_instance = {
        'image': {
            'path': im_path,
            'metatensor': im_tensor,
            'meta_dict': im_meta_dict
        },
        'label': {
            'path': label_path,
            'metatensor': label_tensor,
            'meta_dict': label_meta_dict
        }
    } 

    return reformat_data_instance


def read_jsons(dataset_abs_path:str, exp_data_type:str, fold :Union[str, None]):
        '''
        This function takes the information pertaining to the data which is being used for experimentation.
        
        It permits two use cases, validation and hold-out test sets. 

        Assumes a dataset structure of the following manner: 

        datasets
        |___ dataset_name 
                  |______ ?
                  |______dataset.json (For validation and/or hold-out test set)
                  |______splits.json (For validation)

        ? = Whatever dataset structure you like, this must be represented in the corresponding dataset.json and splits.json files however. We recommend a MSD (or nnU-Net) 
        convention. 

        The corresponding results will be placed in 


        Assumes the following information is stored in a dictionary in the dataset.json file:
            
            Key: ext - File extension type (e.g. nii.gz is only currently supported) 
            
            Key: "training" - List of dictionaries for each image-sample in the training/val set, contains two keys: "image" and "label", each with a relative path from dataset_name 
            to the corresponding data (with no ext)

            Key: "test" - List of dictionaries for each image-sample in the hold out test set, contains two keys: "image" and "label", each with a relative path from dataset_name 
            to the corresponding data (with no ext)

            Key: class_labels - Class label configs in a dictionary format: {"class_1": int, "class_2": int, etc.}

        Assumes the following information is stored in the splits.json file (REQUIRED for validation):
            Dictionary for each fold, with keys "fold_i" with i = 0,1,2,...
                Each fold contains a list of the dictionaries for each image-sample in the fold. Each dict contains two keys: "image" and "label", each with a relative path
                from dataset_name. 

        Inputs:
            dataset_abs_path: The absolute path to the dataset directory
            exp_data_type: The data split type for the experiment (e.g. val, test)
            fold: Optional, in instances where the experiment is done on a validation fold.

        Returns: 
            Returns the list of dictionaries for the paths of the images and labels. 
        '''
        
        overall_data_json_path = os.path.join(dataset_abs_path, 'dataset.json') 
        val_split_path = os.path.join(dataset_abs_path, 'splits.json')

        if exp_data_type.capitalize() == 'Test':
            with open(overall_data_json_path, "r") as f:
                data_store_dict = json.loads(f)
            return data_store_dict['test']
        
        elif exp_data_type.capitalize() == 'Val':
            with open(val_split_path, 'r') as f:
                data_store_dict = json.loads(f)
            return data_store_dict[f"fold_{fold}"]
        else:
            raise KeyError('Invalid experiment data set type selected, must be Test or Val.')