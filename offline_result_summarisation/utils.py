import json
import os
import sys

def extract_config(path, name):
    #Function which extracts configs dicts from json or txt files. Takes the path to the file, and the name of the specific config desired.

    if not os.path.exists(path):
        raise Exception(f'The path {path} was not a valid one. Please check.')    

    #Loading the file:
    with open(path) as f:
        configs_registry = json.load(f)
        config = configs_registry[name]
    return config

def convert_nnunet_task_to_internal_convention(nnunet_task_id, mapping_file_path):
    '''
    inputs: 
    nnunet_task_id: str, the ID of the nnUNet task, not the dataset id! This is still in our own
    convention, but is wrapped under the dataset id (which IS in nnUNet_convention). 
    mapping_file_path: str, path to the JSON file containing the mappings for the given nnUNet dataset
    which will map between nnUNet tasksand our internal convention.

    Example:
    We have Dataset040 which has types of permutations on labels. Then we would have two nnUNet datasets,
    but in the internal convention it would only correspond to two groups of task IDs. This functions
    to provide a map between the two (this is practically just intended for spring cleaning...) 
    
    Function intended to provide pointers between our convention and nnUNet's naming conventions.

    Why: Well, within our dataset-level schema we only differentiate between datasets based on
    permutations to the image-data offline (if we want to enable working with a cache!). This doesn't
    apply to the segmentations however, and we want flexibility to mix/match segmentation tasks without
    duplication. 

    However, nnUNet operates under the convention that each permutation of the data (whether image or
    segmentation) is a separate dataset/task (as these constitute samples for an underlying sample
    distribution). Therefore, if we wanted to map the same image-data to different tasks (based on 
    segmentation permutations) we need to create a pointer for mapping nnUNet_tasks to our internal
    convention. This functions will convert between the two according to the cached mapping (in essence lookup tables).
    '''
    config = extract_config(mapping_file_path, nnunet_task_id)
    #We assert that the config must contain two fields at least:
    #1) The relpath to the internal validation framework's convention for task identification.
    #2) The task id in the validation framework's convention for task identification.
    if config.get('exp_config_relpath', None) is None:
        raise ValueError(f"Config for nnUNet task ID {nnunet_task_id} is missing 'exp_config_relpath'.")
    if config.get('task_id', None) is None:
        raise ValueError(f"Internal validation framework ID correspondence is missing for nnUNet task ID {nnunet_task_id}.")
    if config.get('dataset_name_val_convention', None) is None:
        raise ValueError(f"Config for nnUNet task ID {nnunet_task_id} is missing 'dataset_name_val_convention'.")
    return config 