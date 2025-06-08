# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Some of the functions have been borrowed from the batchgenerators repository's folder operations. We removed some which seem unrequired.
import os
import pickle
import json
import sys 
from typing import List, Union, Optional
import os
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import warnings 

#we classify medio vs non-medio file-types based on whether they store their metadata within the file itself (or whether it would be stored in a separate file)
#we choose to exclusively work with medio file-types in the backend, so any conversion will be performed "out of the loop".
SUPPORTED_DOWNSTREAM_MEDIO_FILE_EXT = ['.nii.gz'] 
# for the upstream (i.e. those which will need to be converted) 
SUPPORTED_UPSTREAM_MEDIO_FILE_EXT = ['.nii.gz']
SUPPORTED_UPSTREAM_NONMEDIO_FILE_EXT = []

#We accumulate all of the supported file types, for the sake of filetype-checking functions, for now this will mostly be used for 
SUPPORTED_FILE_EXT = SUPPORTED_DOWNSTREAM_MEDIO_FILE_EXT + SUPPORTED_UPSTREAM_MEDIO_FILE_EXT + SUPPORTED_UPSTREAM_NONMEDIO_FILE_EXT 

def check_dataset_existence(datasets_path: str, dataset_id: int):
    #We only want to check the existence of datasets in our stack of datasets! To reduce redundancy where possible.
    startswith = "Dataset%03.0d" % dataset_id
    available_datasets = extract_subdirs(datasets_path, prefix=startswith, join=False)
    unique_datasets = np.unique(available_datasets)
    return unique_datasets

def extract_subdirs(folder: str, join: bool = True, prefix: Optional[str] = None, suffix: Optional[str] = None, sort: bool = True) -> List[str]:
    """
    Returns a list of subdirectories in a given folder, optionally filtering by prefix and suffix,
    and optionally sorting the results. Uses os.scandir for efficient directory traversal.

    Parameters:
    - folder: Path to the folder to list subdirectories from.
    - join: Whether to return full paths to subdirectories (if True) or just directory names (if False).
    - prefix: Only include subdirectories that start with this prefix (if provided).
    - suffix: Only include subdirectories that end with this suffix (if provided).
    - sort: Whether to sort the list of subdirectories alphabetically.

    Returns:
    - List of subdirectory paths (or names) meeting the specified criteria.
    """
    subdirectories = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir() and \
               (prefix is None or entry.name.startswith(prefix)) and \
               (suffix is None or entry.name.endswith(suffix)):
                dir_path = entry.path if join else entry.name
                subdirectories.append(dir_path)

    if sort:
        subdirectories.sort()

    return subdirectories


def extract_subfiles(folder: str, join: bool = True, prefix: Optional[str] = None, suffix: Optional[str] = None, sort: bool = True) -> List[str]:
    """
    Returns a list of files in a given folder, optionally filtering by prefix and suffix,
    and optionally sorting the results. Uses os.scandir for efficient directory traversal,
    making it suitable for network drives.

    Parameters:
    - folder: Path to the folder to list files from.
    - join: Whether to return full file paths (if True) or just file names (if False).
    - prefix: Only include files that start with this prefix (if provided).
    - suffix: Only include files that end with this suffix (if provided).
    - sort: Whether to sort the list of files alphabetically.

    Returns:
    - List of file paths (or names) meeting the specified criteria.
    """
    files = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file() and \
               (prefix is None or entry.name.startswith(prefix)) and \
               (suffix is None or entry.name.endswith(suffix)):
                file_path = entry.path if join else entry.name
                files.append(file_path)

    if sort:
        files.sort()

    return files

def extract_nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return extract_subfiles(folder, join=join, sort=sort, suffix='.nii.gz')

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def full_path_splitter(path: str) -> List[str]:
    """
    Replacing the use of a string based splitting function with that of nnU-Net's batchgenerator used method.
    Splits at each separator according to the OS. This is different from os.path.split which only splits at last separator! 
    """
    return path.split(os.sep)

# def file_ext_splitter(filename:str, suffix: str) -> List[str]:
#     """
#     Splits an image filename into the filename & its corresponding extension. Intended to be compatible with file extensions that are not necessarily single suffix.
#     Assumes only that the filename is provided, not the full path. The fullpath splitter is provided for that!
#     """
#     if suffix not in SUPPORTED_DOWNSTREAM_FILE_EXT:
#         raise Exception('Unsupported file extension for processing')
#     return filename.split(f'{suffix}')

def file_ext_splitter(filename:str, suffix:str):

    '''    
    Splits a filename (typically an image filename) into the filename & its corresponding extension. Intended to be compatible with file extensions that are not 
    necessarily single suffix. Assumes only that the filename is provided, not the full path. The fullpath splitter is provided for that!
    
    Possible limitations:

    Missed extensions if the file ends with an unknown or unexpected extension.
    Ambiguity if multiple extensions match (not common if sorted correctly).
    '''
    if suffix not in SUPPORTED_FILE_EXT:
        raise warnings.warn('Unsupported file extension for processing')
    path = Path(filename)
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)], suffix    
    raise ValueError('The filename provided did not have a file extension/type matching the provided suffix.')


def is_supported_filetype(path, ext):
    return bool(ext in SUPPORTED_FILE_EXT)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  
def standard_dataset_json_gen(output_folder: str,
                        annotator_dict: dict, 
                        channel_names: dict,
                        semantic_labels: dict,
                        num_training_cases: int,
                        # num_test_cases: int,
                        file_ext: str,
                        tensorImageSize: str, 
                        reference: str,
                        train_relpaths: dict, 
                        num_test_cases: int = 0,
                        test_relpaths: dict = None,  
                        citation: Union[List[str], str] = None,
                        dataset_name: str = None,
                        release: str = None,
                        description: str = None,
                        # overwrite_image_reader_writer: str = None,
                        license: str = 'Whoever converted this dataset was lazy and didn\'t look it up! I am quite lazy usually...in fact I borrowed some of this code!',
                        converted_by: str = 'Parhom Esmaeili', #"Please enter your name, especially when sharing datasets with others in a common infrastructure!",
                        **kwargs):
    """
    Generates a dataset.json file in the output folder, structure is inherited from the nnu-net convention and adapted for our assumed folder structure. This acts more 
    like a checklist, than anything else. For the non-compulsory fields they're provided as optional parameters.

    
    dataset_name, reference, release, license, description, converted_by: self-explanatory, mostly for completeness
    tensorImageSize: Denotes the number of dimensions for the image array. 3D = Spatial only - single series volume. 4D = multi-series volume.
    numTraining: is used to double check all cases are there!
    numTest: is used to double check!
    
    Why have a distinction if this framework is primarily for validation? Two main reasons, 1) we would like challenge designers to be able to exploit the entire dataset 
    if they desire (or people trying to perform analyses without access to the test datasets....... :( ). 
    2) model design process benefits from the use of a better validation mechanism (so it helps those partaking in challenges/designing their own
    algorithms more generally).

    annotators: a configuration dict denoting the general descriptor for the annotation generation mechanism
        annotators:{   
            "annotator_name": (can be as simple as "annotator_1" or anything really, we don't use the string for anything other than pathing)
                {
                    "annotator_id": str(int) denoting the numeric code for the annotator for downstream use. Starts always at 1, with increasing integer values.
                    "annotation_protocol": Optional[str] (function input): Some brief descriptor (typically a URL or reference, or maybe a string) denoting the annotation protocol which g
                    enerated the annotations for a given annotator.

                    The protocol is also provided for instances where synthetic datasets have been generated for evaluating multi-annotator algorithms.
                }
        }
    
    semantic_classes: For a general panoptic segmentation task, this refers to the semantic (words) classes of the targets (for both stuff and things) 

        Example:
        {
            'background': {   
                "id": 0,
                "optional": true,
                "semantic_type": stuff
                }
            'left atrium': {   
                "id": 1,
                "optional": true,
                "semantic_type": stuff
                }
            'some other label': {   
                "id": 0,
                "optional": true,
                "semantic_type": stuff
                }
        }

        id = semantic class integer code id. We always expect consecutive integers for labels and also expects 0 to be background (the most general purpose "stuff" 
        class in medical image which designates what can be completely "ignored"!).
        optional = boolean which determines whether the semantic class is optional for any given sample (i.e. whether it is strictly required). True = optional/non-strict.
        semantic_type = the type of segmentation target, stuff or thing.

        Typically, for the vast majority of datasets which are purely semantic segmentation, it will follow this exact structure. The optional case is mostly just a fall-back.

    channel_names:
        Channel name dict which maps an image series/channel to the numeric index representing the channel, example:
        {
            'T1':"0",
            'CT':"1"
        }
    
        
    file_ext: needed for finding the files correctly. For now we assume that the file endings must match between images and segmentations! This property might be disc-
    ontinued if segmentations are converted to nrrd files for compressing the space taken up by panoptic/instance seg files.......

    "training": a dictionary of cases with their corresponding relpaths for the image data, and the segmentation annotations.
    "test": same thing as the training (if available). Typically the test labels will not be available for algorithm designers, but sometimes test images too! 

    kwargs: whatever you put here will be placed in the dataset.json as well

    ================================================================================================================================

    A temporary convention being used for now to describe the training/test dictionaries:
     
    For now we will be assuming a nifti-like format (or any format which does not provide vector-valued tensors like nrrd files can):

    For image data:

    Case/Sample
    |_____Channel files

    For annotations:

    Case/Sample
    |___Annotator
        |_____Semantic class (We assume this much is consistent otherwise what is the point of this framework! Harmonising semantic classes is a different thing and 
                              will be left to the Vision-Language modellers of the world)
            |_______ Instance ID files

    """

    dataset_json = dict() 

    #Optional fields:
    # Dataset_name, release, citation, license, description: self-explanatory, mostly for completeness
    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    # if reference is not None:
    if release is not None:
        dataset_json['release'] = release
    if citation is not None:
        dataset_json['citation'] = citation 
    if description is not None:
        dataset_json['description'] = description
    
    #Obligatory fields: We need a reference!
    dataset_json['reference'] = reference
    dataset_json['licence'] = license
    dataset_json['converted_by'] = converted_by
    # tensorImageSize: Denotes the "real" number of dimensions for the image array. 3D = Spatial only - single series volume. 4D = multi-series volume.
    # Intended to distinguish between the use of single-series volumetric data, and multi-series data in the circumstances where the image files are presented as 3D volumes.
    
    dataset_json['tensorImageSize'] = tensorImageSize
    
    # numTraining: is used to double check all cases are there in the relpaths!
    # numTest: is used to double check all cases are there in the relpaths!
    dataset_json['numTraining'] = num_training_cases
    dataset_json['numTest'] = num_test_cases
    
    #annotators config dict, need to check whether a description is available (i.e. non-nonetype) for the annotation protocol, otherwise provide reference and 
    # hope for the best!? :
    for annotator_conf in annotator_dict.values():
        if annotator_conf['annotation_protocol'] is None:
            annotator_conf['annotation_protocol'] = dataset_json['reference']
    
    dataset_json['annotators'] = annotator_dict

    #semantic_class configs need str(int) as values for the id. For the semantic type it must be stuff or thing, and for the optional it must be a bool.
    for k, sem_conf in semantic_labels.items():
        #conf id checking.
        if isinstance(sem_conf['id'], str):
            if not sem_conf['id'].isdigit():
                raise Exception('Semantic class id codes should only be int values') #It will automatically flag if this value is not an int.
        elif isinstance(sem_conf['id'], int):
            sem_conf['id'] = str(sem_conf['id'])
        else:
            raise Exception('Cannot have a float type integer code for the semantic class code representation.')
        
        #Checking that background semantic class should be represented by 0, this is an almost universal semantic class.
        if int(sem_conf['id']) == 0:
            assert k.title() == 'Background' #Reasonable assumption/standard.
        
        #checking the optional, it MUST be a bool, don't even bother checking otherwise.
        if not isinstance(sem_conf['optional'], bool):
            raise Exception('The optional label tag designating whether a semantic class can be optional for any given sample was not a bool..')
        
        if not isinstance(sem_conf['semantic_type'], str):
            raise Exception('The semantic type parameter must be a string')
        else:
            #checking that it is either stuff or thing!
            if sem_conf['semantic_type'].title() not in ['Stuff', 'Thing']:
                raise Exception('The semantic types must be either stuff or thing, it cannot be anything else. Descriptors are required for downstream application')
    #Checking that the integer codes for the semantic classes are consecutive natural numbers.
    
    id_vals = sorted([int(sem_conf["id"]) for sem_conf in semantic_labels.values()])
    assert id_vals == list(range(len(id_vals))), "Values must be consecutive integers starting from 0"

    dataset_json['semantic_classes'] = semantic_labels
    
    # channel names need strings as values. #Channel_names is the most generic term for multi-"series" (e.g. mpMRI data, or constrast-phase, or multi-modal etc.)
    for k, v in channel_names.items():
        if not isinstance(v, str):
            channel_names[k] = str(v)
    dataset_json['channel_names'] = channel_names 

    #checking that the file extension is one of the supported types.
    if file_ext not in SUPPORTED_DOWNSTREAM_MEDIO_FILE_EXT:
        raise Exception('The filetype is not yet supported downstream!')
    else:
        dataset_json['file_ext'] = file_ext 

    
    #Now we check through and ensure that the number of training and test cases are consistent with the quantity designated above.
    #We make a minimal amount of additional sanity checks just on the relpaths themselves, separate from the structure validation scripts. 
    # We do not necessarily presume a given folder structure, for the segmentations, but we can assume a consistent nnu-net like structure for the image data itself.

    #We do assert a high-level structure for the relpaths: case
    #                                                        |----> images, labels  ----------
    #                                                                  |----> channels,      |---> annotators

    #Some of the checks we perform here:    
    # 1) There are as many relpath descriptors as the "numXXXX" quantity - consistent across train and test
    # 2) Each sample/case must have at least one channel for image data AND at most N channels. Where N reflects the number of channels in the channels_dict. 
        #consistent across train and test. 
    # 3) For the train: Each sample/case has at least 1 annotator subdict which is not empty,
    #    For the test: Need not have annotation as sometimes that data is not provided to the designer of an algorithm!).

    #Checking that the rel_paths, if required, match the dictionary structure.
    if not isinstance(train_relpaths, dict):
        raise Exception('The relpaths are assumed to be in a nested dictionary format')
    if len(train_relpaths) != num_training_cases:
        raise Exception('Mismatch between the number of training cases, and the number of relpaths for saving to the dataset.json')
    
    if num_test_cases == 0:
        if test_relpaths is not None: 
            raise Exception('The test relpaths must be none-type if there were no hold-out test samples explicitly being held out.')
    else:
        if not isinstance(test_relpaths, dict):
            raise Exception('The test relpaths, if being provided are assumed to be in a nested dictionary format')
        if len(test_relpaths) != num_test_cases:
            raise Exception('Mismatch between the number of test cases, and the number of relpaths for saving to the dataset.json')

    #Iterating through and performing basic sanity-checks on the relpaths.
    for case in train_relpaths.values():
        if all([isinstance(v, dict) for v in case.values()]):
            if case['labels'] == {} or not any([v != {} for v in case['labels'].values()]): #checking that at least one subdict is non-empty. 
                raise Exception('Training cases must have some labels!')
        else:
            raise Exception('The key:values of each case must be composed of a datatype:dictionary format, otherwise we cannot store the relpaths compactly.')
        #We iterate over the subdicts corresponding to each of the cases for validating, we check that the number of image channels is non-breaking.
        if len(case['images']) < 1 or len(case['images']) > len(channel_names):
            raise Exception('The image data either had no channels, or more channels than the upper bound.')
    
    dataset_json.update({
        'train': train_relpaths})
    

    #Now for the "hold-out test data"
    if test_relpaths is not None: #The test-data may not always be available. 
        for case in test_relpaths.values():
            #We iterate over the subdicts corresponding to each of the cases for validating.

            #Slightly modified from above, as we do not necessarily assume that the test set should have labels (so algorithm designers can't work within challenge constraints without
            # pain-staking processing).
            if not all([isinstance(v, dict) for v in case.values()]):
                raise Exception('The key:values of each case must be composed of a datatype:dictionary format, otherwise we cannot store the relpaths compactly.')
            #checking that a non-breaking quantity of image channels are provided.
            if len(case['images']) < 1 or len(case['images']) > len(channel_names):
                raise Exception('The image data either had no channels, or more channels than the upper bound.')
            
        dataset_json.update({    
            'test': test_relpaths       
            })


    #Now adding any additional kwargs which may be highly dataset specific.
    dataset_json.update(kwargs)

    save_json(dataset_json, os.path.join(output_folder, 'dataset.json'), sort_keys=False)