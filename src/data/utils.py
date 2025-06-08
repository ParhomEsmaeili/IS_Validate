from monai.transforms import Compose, LoadImaged, Orientationd, EnsureChannelFirstd, EnsureTyped
from monai.data import Dataset, MetaTensor, DataLoader 
import json 
import torch  
import numpy as np
from typing import Union, Optional 
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #Adding the parent directory to the path so that we can import utils from there.
from utils.dict_utils import extractor #This is a custom utility function for extracting dicts.
import copy 
import logging
import pathlib
import warnings 

# logger = logging.getLogger(__name__)

def init_task_cases(dataset_dir:str, exp_task_configs:dict): #, file_ext:str):
    '''
    This function creates a dictionary of the task cases, which is then used, alongside some dataloader utilities, in order to create
    a dataset object which can be used for iterating through the cases in the task.
    '''

    #Reading the dataset.json file which will contain the base information about the dataset as initially formulated offline.
    try:
        dataset_json_path = os.path.join(dataset_dir, 'dataset.json')
        #Reading the json file. 
        with open(dataset_json_path) as f:
            ds_configs = json.load(f)
    except:
        dataset_txt_path = os.path.join(dataset_dir, 'dataset.txt')
        #Reading the txt file. 
        with open(dataset_txt_path) as f:
            ds_configs = json.load(f)

    #Reading the dataset_split.json file which will describe the cases which will be used for the experiment.

    try:
        dataset_split_path = os.path.join(dataset_dir, 'dataset_split.json')
        #Reading the json file.
        with open(dataset_split_path) as f:
            data_sampling_configs = json.load(f) 
    except:
        dataset_txt_path = os.path.join(dataset_dir, 'dataset_split.txt')
        #Reading the txt file.
        with open(dataset_txt_path) as f:
            data_sampling_configs = json.load(f) 

    #Extracting the case list according to the list provided in the sampling category description in the task configs:
    split_dict_path = ['sampling'] + exp_task_configs['data_sampling']['sample_group_category'] 
    split_dict_path = tuple(split_dict_path) if isinstance(split_dict_path, list) else split_dict_path
    #We will use the extractor function to extract the case list from the data_sampling_configs.
    sample_category = exp_task_configs['data_sampling']['sample_group_category'][0]
    if not isinstance(sample_category, str):
        raise TypeError('The sample_group_category must be a string denoting the sampling category, e.g. "all_" or "kfold_".')
    if not sample_category.startswith(('all_', 'kfold_')):
        raise NotImplementedError('The sample_group_category must start with "all_" or "kfold_" for now. Other sampling categories are not yet supported.')
    split_metadata_path = ('sampling', sample_category, 'meta')

    #Now we will extract the case list and the metadata from the data_sampling_configs
    case_list = extractor(data_sampling_configs, split_dict_path) 
    if not isinstance(case_list, list) or not case_list:
        #If the case list is not a list or is empty, we raise an error.
        raise TypeError('The case list must be a list of strings denoting each case folder to be used.')
    split_metadata = extractor(data_sampling_configs, split_metadata_path)
    if not isinstance(split_metadata, dict) or not split_metadata:
        #If the split metadata is not a dict or is empty, we raise an error.
        raise TypeError('The split metadata must be a non-empty dictionary containing information about the split.')
    
    #We will put a sanity check here to ensure that the user is not attempting to provide a case list for which the dataset.json could 
    #quite possible not have produced any reasonable cases. 
    #
    # I.e., in the example where the dataset.json has numTest=0 but the user is attempting to use something sampled from test set for the 
    # experiment, we will raise an error. Moreover, we will check that the cases all have corresponding annotations, because for now
    # we do not support cases without annotations.
    if ds_configs[f"num{split_metadata['split'].capitalize()}"] == 0:
        raise ValueError(f'The dataset.json file does not contain any cases for the {split_metadata["split"]} set, but the experiment is attempting to use a sample from this set')
        #TODO: We need to provide a better name for the split metadata key, as this is not very descriptive. It can easily be confused
        #with the split within the subset, while for now it is referring to the split of the dataset as a whole, i.e. train, test.
    if not all(['label' in ds_configs[split_metadata['split']][case].keys() for case in case_list]):
        raise ValueError(f'The dataset.json file does not contain labels for all cases in the {split_metadata["split"]} set, but the mechanisms provided currently require annotations.')


    #We now put some type checks to ensure that the quantity of cases is correct, corresponding to the number of cases in the dataset.json file,
    #and the sampling category, which provides a description of the mechanism for sampling the cases. This is the final line of sanity
    #checking that the sampling was done correctly offline. E.g., k-fold must have a specific structure. 

    #We extract the specific source for the data (i.e. train set or test set) from the metadata in the data_splits.json file, in order to 
    # cross reference against the dataset.json file. 

    if exp_task_configs['data_sampling']['sample_group_category'].beginswith('all_'):
        #If the sampling category begins with 'all_', then we assume that all cases are to be used.
        if not len(case_list) == ds_configs[f"num{split_metadata['split'].capitalize()}"]:
            expected_num = ds_configs[f'num{split_metadata["split"].capitalize()}']
            raise ValueError(
                f'The number of cases in the dataset.json file does not match the number of cases in the {split_metadata["split"]} set, '
                f'expected {expected_num} but got {len(case_list)}'
            )
    elif exp_task_configs['data_sampling']['sample_group_category'].startswith('kfold_'):
        #If the sampling category is kfold then we assume that the cases are to be sampled from a k-fold cross-validation setup. 
        # We will check that the number of cases in the fold is feasible. i.e., that it is equivalent to the floor division of the 
        # number of cases in the dataset.json file by the number of folds. 
        num_folds = int(split_metadata['k_folds']) 
        expected_num = ds_configs[f'num{split_metadata["split"].capitalize()}'] // num_folds
        if not len(case_list) == expected_num:
            raise ValueError(
                f'The number of cases in the dataset.json file does not match the number of cases in the {split_metadata["split"]} set for the k-fold cross-validation, '
                f'expected {expected_num} but got {len(case_list)}'
            )
    else:
        #We do not yet support any other sampling categories, so we raise an error.
        raise NotImplementedError('The sample category must be either "all_" or "kfold_" for now. Other sampling categories are not yet supported.')

    #Now that we know which cases we are using, we can extract some case specific information with respect to the task at hand.


    #Extracting information about the task with respect to the segmentation problem. 
    orig_semantic_classes_dict = ds_configs.get('semantic_classes') #We do not put a failsafe because this is a required field.
    
    if not isinstance(orig_semantic_classes_dict, dict):
        raise Exception('Semantic labels must be in a dictionary mapping format.')
    else:
        #We need to check that the keys describing the semantic classes are strings.
        if not all([isinstance(i, str) for i in orig_semantic_classes_dict.keys()]):
            raise TypeError('The keys in the semantic classes dict must be a str denoting the text semantic info about the class')
        #We need to check whether the ids are all integers
        if not all([isinstance(i, int) for i in orig_semantic_classes_dict.values()]):
            raise TypeError('The semantic class-integer codes must be an int')
        if any([i['semantic_type'].capitalize() == 'Thing' for i in orig_semantic_classes_dict.values()]):
            raise NotImplementedError('The semantic type for the default input dataset is a "thing", but we do not yet provide proper handling for panoptic datasets.')
    
    
    #Here we will extract some more specific information for the more complex configurations of data, e.g. multi-channel images, 
    # multi-annotator/annotator specific setups, the selection of specific segmentation instances if at all, etc. Currently we will 
    # assume that all of these parameters should be lists of length 1, as we do not yet support multi-channel or multi-annotator setups.
    
    return config_labels_dict, dataloader_generator(datalist=datalist)


def create_cases(recursive_dict):

    #A function which will create the default structure of the cases, which will be passed through to the dataloader for case by case
    #processing downstream. 



    #Determining the actual filepaths for the task at hand: Modifying case-list to include abspath and file extensions

    #Extracting the os name.
    os_name = os.name 

    for case_instance in case_list:

        if os_name == 'nt':
            #Windows 
            #Expects that the relative path provided is in windows format.
            im_path = os.path.join(dataset_dir, str(pathlib.PureWindowsPath(data_instance['image'])))
            lb_path = os.path.join(dataset_dir, str(pathlib.PureWindowsPath(data_instance['label'])))
        elif os_name == 'posix':
            #Posixtype 
            #Expects that the relative path provided is in posix format.
            im_path = os.path.join(dataset_dir, str(pathlib.PurePosixPath(data_instance['image'])))
            lb_path = os.path.join(dataset_dir, str(pathlib.PurePosixPath(data_instance['label'])))
        else:
            raise Exception('OS not supported.')

        data_instance['image'] = im_path + file_ext
        data_instance['label'] = lb_path + file_ext  


    return case_list 






def dataloader_generator(datalist):
    '''
    This function handles the construction of a dataset object for iterating through.
    '''
    load_transforms = [
        LoadImaged(keys=['image', 'label'], reader="ITKReader", image_only=True),
        EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        EnsureTyped(keys=['image', 'label'], dtype=[torch.float64, torch.int64]),
    ]
    dataset = Dataset(datalist, Compose(load_transforms))
    # return DataLoader(dataset=dataset, batch_size=1, num_workers=1)

    #We switch from DataLoader to just using the Dataset as we do not want our meta dictionary to be endowed with any additional structure induced by the dataloader's
    #support for batch sizes > 1 at each iteration/callback. 
    return dataset  

def iterate_dataloader_check(data_instance):
    
    if not isinstance(data_instance, dict):
        raise TypeError('Data loader requires dictionary based transforms to be passed through the dataloader')
            

    if isinstance(data_instance['image'], MetaTensor) and isinstance(data_instance['label'], MetaTensor):
        try:
            im_meta_dict = data_instance['image'].meta #['image_meta_dict']
            label_meta_dict = data_instance['label'].meta #['label_meta_dict']
        except:
            raise Exception('The loaded data instance does not contain a meta dictionary')

        #We assert that the data must be single modality for our current application!
        if int(im_meta_dict['pixdim[4]']) > 1:
            raise ValueError('This application only supports single modality implementations.')
        if int(label_meta_dict['pixdim[4]']) > 1:
            raise ValueError('This application only supports single modality implementations')
    else:
        raise TypeError("The loaded image and label data must be a MetaTensor")

    



def data_instance_reformat(data_instance:dict):
     
    '''
    This function reformats the data instance from the output of the dataset generator's load transforms into the 
    data_instance reformat expected by the pseudo-ui.
    '''
    if not isinstance(data_instance, dict) or not data_instance:
        raise Exception('Data instance is assumed to be a non-empty dictionary format..')

    #We extract the paths from the meta info:

    im_path = copy.deepcopy(data_instance['image'].meta['filename_or_obj'])
    label_path = copy.deepcopy(data_instance['label'].meta['filename_or_obj'])

    if not os.path.exists(im_path) or not os.path.exists(label_path):
        raise Exception('One of the paths does not exist! Or the full absolute path was not provided to the dataset constructor')
    
    im_tensor = copy.deepcopy(data_instance['image'])
    label_tensor = copy.deepcopy(data_instance['label'])

    if not isinstance(im_tensor, MetaTensor): #or not isinstance(im_tensor, torch.Tensor):
        raise TypeError('Image tensor was not a MONAI meta-tensor.')
    if not isinstance(label_tensor, MetaTensor): #or not isinstance(label_tensor, torch.Tensor): 
        raise TypeError('Label tensor was not a MONAI meta-tensor.')
    
    #Wiping all of the metadictionary outside of the affine and original affine keys. The user should not, and does not, require any of this other information! Bye bye!
    retained_keys = ('original_affine', 'affine')

    im_tensor.meta = {key:val for key, val in im_tensor.meta.items() if key in retained_keys}
    label_tensor.meta = {key:val for key, val in label_tensor.meta.items() if key in retained_keys}    

    
    # im_meta_dict = copy.deepcopy(data_instance['image'].meta)
    # label_meta_dict = copy.deepcopy(data_instance['label'].meta)

    if not isinstance(im_tensor.meta, dict) or not im_tensor.meta:
        raise Exception('The image meta_dict must be a non-empty dictionary')
    
    if not isinstance(label_tensor.meta, dict) or not label_tensor.meta:
        raise Exception('The label meta_dict must be a non-empty dictionary') 

    #Checking that the meta dict contains the affine arrays required.

    #First copying the meta dict to keep it separate from the metatensor. 

    im_meta_dict = copy.deepcopy(im_tensor.meta)
    label_meta_dict = copy.deepcopy(label_tensor.meta)

    original_affine_key = 'original_affine'
    current_affine_key = 'affine'

    def affine_checker(meta_dict:dict, affine_key: str):
        if isinstance(meta_dict[affine_key], np.ndarray):
            warnings.warn(f'The meta_dicts key: {affine_key} was a numpy array, converting to torch tensor.')
            meta_dict[affine_key] = torch.from_numpy(meta_dict[affine_key])
            return meta_dict 
        elif isinstance(meta_dict[affine_key], torch.Tensor):
            return meta_dict
        else:
            raise TypeError('The affine info must be presented as a numpy or torch tensor.')
    #Running the affine checker for the original and current affine; for both the image and the label
    im_meta_dict = affine_checker(im_meta_dict, original_affine_key)
    im_meta_dict = affine_checker(im_meta_dict, current_affine_key)

    label_meta_dict = affine_checker(label_meta_dict, original_affine_key)
    label_meta_dict = affine_checker(label_meta_dict, current_affine_key)
    

    #Also applying this to the .meta attribute of the metatensors through reassignment.
    im_tensor.meta = copy.deepcopy(im_meta_dict)
    label_tensor.meta = copy.deepcopy(label_meta_dict)
    


    #Now populating the reformatted data instance:

    reformat_data_instance = {
        'image': {
            # 'path': im_path,
            'metatensor': im_tensor,
            'meta_dict': im_meta_dict
        },
        'label': {
            # 'path': label_path,
            'metatensor': label_tensor,
            'meta_dict': label_meta_dict
        }
    } 
    
    if os.path.split(im_path)[1].split('.')[0] != os.path.split(label_path)[1].split('.')[0]:
        raise Exception('The name for the provided image and label needs to be the same!')
    

    patient_name = os.path.split(im_path)[1].split('.')[0]

    return reformat_data_instance, patient_name


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