from monai.transforms import Compose, LoadImaged, Orientationd, EnsureChannelFirstd, EnsureTyped
from monai.data import Dataset, MetaTensor, DataLoader 
import json 
import torch  
import numpy as np
from typing import Union, Optional, Sequence
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #Adding the parent directory to the path so that we can import utils from there.
from general_utils.dict_utils import extractor #This is a custom utility function for extracting dicts.
import copy 
import logging
import pathlib
import warnings 
from skimage.measure import label as cc_label
from version_handling import monai_version 
import gc
# logger = logging.getLogger(__name__)

def init_task_cases(dataset_dir:str, exp_task_configs:dict): #, file_ext:str):
    '''
    This function creates a dictionary of the task cases, which is then used, alongside some dataloader utilities, in order to create
    a dataset object which can be used for iterating through the cases in the task.

    inputs: 
    dataset_dir (path to the dataset directory)
    dataloading_type (currently only designates between a basic dataloading, i.e. just doing a very basic loading, fusion of semantic class 
    labels and orientation into the RAS convention, and a non-basic dataloading (which will inspect the dataloader transforms in the task configs
    dictionary))
    exp_task_configs (a dictionary which contains information about the description of the task for loading the cases/building them.)
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
    sampling_metadata_path = ('sampling', sample_category, 'meta')

    #Now we will extract the case list and the metadata from the data_sampling_configs
    case_list = extractor(data_sampling_configs, split_dict_path) 
    
    # Check that there are no repeated cases in the case list, even though the dataset constructor permits this as it only 
    #requires the use of a list and not a dict construction in the input argument.
    if len(case_list) != len(set(case_list)):
        raise NotImplementedError('Duplicate cases found in the case list. Repeats are currently not supported for the interpretation of downstream calculations'
        '.Caution is required.')
    if not isinstance(case_list, list) or not case_list:
        #If the case list is not a list or is empty, we raise an error.
        raise TypeError('The case list must be a list of strings denoting each case folder to be used.')

    #Now we extract the metadata.
    sampling_metadata = extractor(data_sampling_configs, sampling_metadata_path)
    if not isinstance(sampling_metadata, dict) or not sampling_metadata:
        #If the split metadata is not a dict or is empty, we raise an error.
        raise TypeError('The split metadata must be a non-empty dictionary containing information about the split.')
    
    #We will put a sanity check here to ensure that the user is not attempting to provide a case list for which the dataset.json could 
    #quite possible not have produced any reasonable cases. 
    #
    # I.e., in the example where the dataset.json has numTest=0 but the user is attempting to use something sampled from test set for the 
    # experiment, we will raise an error. Moreover, we will check that the cases all have corresponding annotations, because for now
    # we do not support cases without annotations.
    if ds_configs[f"num{sampling_metadata['split'].capitalize()}"] == 0:
        raise ValueError(f'The dataset.json file does not contain any cases for the {sampling_metadata["split"]} set, but the experiment is attempting to use a sample from this set')
        #TODO: We need to provide a better name for the split metadata key, as this is not very descriptive. It can easily be confused
        #with the split within the subset, while for now it is referring to the split of the dataset as a whole, i.e. train, test.
    if not all(['labels' in ds_configs[sampling_metadata['split']][case].keys() for case in case_list]):
        raise ValueError(f'The dataset.json file does not contain labels for all cases in the {sampling_metadata["split"]} set, but the mechanisms provided currently require annotations.')


    #We now put some type checks to ensure that the quantity of cases is correct, corresponding to the number of cases in the dataset.json file,
    #and the sampling category, which provides a description of the mechanism for sampling the cases. This is the final line of sanity
    #checking that the sampling was done correctly offline. E.g., k-fold must have a specific structure. 

    #We extract the specific source for the data (i.e. train set or test set) from the metadata in the data_splits.json file, in order to 
    # cross reference against the dataset.json file. 

    if sample_category.startswith('all_'):
        #If the sampling category begins with 'all_', then we assume that all cases are to be used.
        if not len(case_list) == ds_configs[f"num{sampling_metadata['split'].capitalize()}"]:
            expected_num = ds_configs[f'num{sampling_metadata["split"].capitalize()}']
            raise ValueError(
                f'The number of cases in the dataset.json file does not match the number of cases in the {sampling_metadata["split"]} set, '
                f'expected {expected_num} but got {len(case_list)}'
            )
    elif sample_category.startswith('kfold_'):
        #If the sampling category is kfold then we assume that the cases are to be sampled from a k-fold cross-validation setup. 
        # We will check that the number of cases in the fold is feasible. i.e., that it is equivalent to the floor division of the 
        # number of cases in the dataset.json file by the number of folds. 
        num_folds = int(sampling_metadata['k_folds']) 
        expected_num = ds_configs[f'num{sampling_metadata["split"].capitalize()}'] // num_folds
        if not len(case_list) == expected_num:
            raise ValueError(
                f'The number of cases in the dataset.json file does not match the number of cases in the {sampling_metadata["split"]} set for the k-fold cross-validation, '
                f'expected {expected_num} but got {len(case_list)}'
            )
    else:
        #We do not yet support any other sampling categories, so we raise an error.
        raise NotImplementedError('The sample category must be either "all_" or "kfold_" for now. Other sampling categories are not yet supported.')
        #Probably have already raised this error above, but just in case we will raise it here as well.



    #Now that we know which cases we are using, we can extract some case specific information with respect to the task at hand.
    #Extracting information about the task with respect to the segmentation problem. 
    orig_semantic_classes_dict = ds_configs.get('semantic_classes') #We do not put a failsafe because this is a required field.
    
    if not isinstance(orig_semantic_classes_dict, dict):
        raise Exception('Semantic labels must be in a dictionary mapping format.')
    else:
        #We need to check that the keys describing the semantic classes are strings.
        if not all([isinstance(i, str) for i in orig_semantic_classes_dict.keys()]):
            raise TypeError('The keys in the semantic classes dict must be a str denoting the text semantic info about the class')
        #For now we will assume that the semantic types are just going to be stuff, so lets raise an error if the default dataset has any thing classes.
        #This is because we do not yet support panoptic datasets, and the default dataset is assumed to be a panoptic dataset.
        if any([i['semantic_type'].capitalize() == 'Thing' for i in orig_semantic_classes_dict.values()]):
            raise NotImplementedError('The semantic type for the default input dataset is a "thing", but we do not yet provide proper handling for panoptic datasets.')
        #We will convert the semantic class ids to integers, as they are currently strings. They are stored as a field of a subdict, so we
        #will iterate through the semantic classes and convert the ids to integers.
        for key, value in orig_semantic_classes_dict.items():
            if isinstance(value['id'], str):
                try:
                    value['id'] = int(value['id'])
                    if value['id'] < 0:
                        raise ValueError(f'The semantic class id for {key} must be a non-negative integer.')
                    if key.capitalize() == 'Background':
                        if value['id'] != 0:
                            raise ValueError(f'The semantic class id for background must be 0, this is our convention.')
                    else:
                        if value['id'] == 0:
                            raise ValueError(f'The semantic class id for non-background semantic class: {key} must be a non-zero integer, this is our convention.')
                except:
                    raise Exception('Unknown error was flagged in checking the semantic class labels, please check.')
            elif not isinstance(value['id'], str):
                if not isinstance(value['id'], int): 
                    raise TypeError(f'The semantic class id for {key} must be an integer, if it was not a str. Highly recommended to use the original convention!')
            orig_semantic_classes_dict[key] = value
        #Just checking that the semantic_class_ids have no duplicates, that they are all successive integers (i.e., 0,1,2)
        if len(set([i['id'] for i in orig_semantic_classes_dict.values()])) != len(orig_semantic_classes_dict):
            raise ValueError('The semantic class ids must be unique integers, but there are duplicates in the provided semantic classes.')
        if set([i['id'] for i in orig_semantic_classes_dict.values()]) != set(range(len(orig_semantic_classes_dict))):
            raise ValueError('The semantic class ids must be successive integers starting from 0, but the provided semantic classes do not match this requirement.')
        
        #Creating a dictionary for mapping the semantic_classes string descriptors from the original descriptors to the 
        #descriptors used in the task. This is used for merging the semantic classes with the task specific labels.
        if 'semantic_class_mapping' in exp_task_configs['data_transforms'].keys():
            #In this case we will need to generate a mapping between descriptors for the downstream loader.
            semantic_class_mapping = exp_task_configs['data_transforms']['semantic_class_mapping']
            config_labels_dict = {k:i for i, k in enumerate(semantic_class_mapping.keys())}
        else:
            warnings.warn('We revert to the default semantic classes, but take care that this is intended.')
            #If the semantic_class_mapping is not provided, we will just use the original semantic classes as the descriptor/integer code.
            semantic_class_mapping = {k:[k] for k in orig_semantic_classes_dict.keys()} #We create a one-to-one map to itself.
            config_labels_dict = {k:v['id'] for k,v in orig_semantic_classes_dict.items()} 
        
        #  We need to check that all semantic classes from the dataset.json are present in the semantic_class_mapping,
        # and that each original semantic class appears only once in the mapping (no duplicates or missing classes).
        mapped_classes = set()
        for mapped_keys in semantic_class_mapping.values():
            if not isinstance(mapped_keys, list) or any([sem_class not in orig_semantic_classes_dict.keys() for sem_class in mapped_keys]):
                raise TypeError("Each value in semantic_class_mapping must be a list of original semantic class names.")
            #Now we will check whether a given class has already been mapped, i.e. whether it has been mapped to multiple diff
            #downstream merged semantic classes.
            for orig_class in mapped_keys:
                if orig_class in mapped_classes:
                    raise ValueError(f"Semantic class '{orig_class}' from dataset.json is mapped more than once in semantic_class_mapping.")
                mapped_classes.add(orig_class)
        #cross-referencing to make sure we didn't have any unaccounted for semantic classes.
        orig_classes_set = set(orig_semantic_classes_dict.keys())
        if mapped_classes != orig_classes_set:
            missing = orig_classes_set - mapped_classes
            extra = mapped_classes - orig_classes_set
            msg = []
            if missing:
                msg.append(f"Unaccounted for semantic classes in mapping: {missing}")
            if extra:
                msg.append(f"Unknown semantic classes in mapping: {extra}")
            raise ValueError(" ".join(msg))
    
        #TODO: Integration of panoptic datasets, i.e. those with "thing" classes, is required. We have made a lot of simplifying
        #assumptions for this implementation with multi-class semantic seg.
        
        if len(config_labels_dict) != 2:
            raise NotImplementedError('Hardcoded exception, downstream apps are only designed to handle binary semantic segmentation problems.')
    
    case_list, image_keys, label_keys = create_case_list(
        dataset_dir=dataset_dir, 
        ds_configs=ds_configs,
        sampling_metadata=sampling_metadata,
        exp_task_configs=exp_task_configs,
        case_list=case_list)

    #Create a temporary transforms configs dictionary. It defines the set of parameters required for any downstream function
    #in the dataloaders which is non-default (i.e. loading to RAS orientation, etc.)

    #Almost always going to be required, for now anyways:
    # Semantic class mapping (i.e. mapping into the task domain for the loaded annotations)
    # image structure mapping (i.e. what was the mechanism through which we generated the keys for the image-related fields.
    # annotation_structure_mapping (i.e. what was the mechanism through which we generated the keys for the annotation-related fields.

    #These are all required, for now, in order to merge them into structures which can be used downstream.
    transforms_configs = {
        #Hardcoding some of the inputs for now.... TODO: Tidy this up later in a more abstracted capacity.
        'semantic_class_mapping':semantic_class_mapping,
        'config_labels_dict': config_labels_dict, 
        'image_struct_mapping': extractor(exp_task_configs, ('data_sampling', 'image_conf')),
        'annotation_struct_mapping':extractor(exp_task_configs, ('data_sampling', 'annotation_conf'))
    }

    #Now we add any other non-semantic class mapping transforms, i.e. the more atypical ones (with respect to the current
    # common conventions). This is just a temporary hacky fix until we build a proper infrastructure for the dataloading pipeline.
    transforms_configs.update({'non_standard_transfs':{k:v for k,v in exp_task_configs['data_transforms'].items() if k != 'semantic_class_mapping'}})

    return config_labels_dict, dataloader_generator(case_list=case_list, image_keys=image_keys, label_keys=label_keys, transforms_configs=transforms_configs)

def create_filepath(dataset_dir, relpath):
    #Determining the actual abspath filepaths for the task at hand. 

    #Extracting the os name.
    os_name = os.name 
    if os_name == 'nt':
        #Windows 
        #Expects that the relative path provided is in windows format.
        filepath = os.path.join(dataset_dir, str(pathlib.PureWindowsPath(relpath))) 
    elif os_name == 'posix':
        #Posixtype 
        #Expects that the relative path provided is in posix format.
        filepath = os.path.join(dataset_dir, str(pathlib.PurePosixPath(relpath)))
    else:
        raise Exception('OS not supported.')


    if not os.path.isfile(filepath):
        raise ValueError('The generated filepath was not found for a file which is to be loaded downstream.')
    else:
        return filepath  

def create_case_list(
        dataset_dir, 
        ds_configs,
        sampling_metadata,
        exp_task_configs, 
        case_list):
    '''
    Function which creates the dictionary of cases which will be required for passing through to the dataloader.

    Takes input arguments:
    
    dataset_dir: The absolute path to the directory which contains everything relevant to the given dataset in the experiment.
    
    ds_configs: The loaded dataset.json from the give dataset_dir, which will contain all of the relevant information for pathing
    and describing the default configuration of the dataset.
    
    sampling_metadata: A dictionary containing the metadata which describes the cases being sampled. 

    exp_task_configs: A dictionary containing all of the relevant information which could be required for describing the task being performed
    in the experiment. 

    case_list: The list of cases which will need to be constructed according to the configurations of the task.
    
    Outputs:

    A dictionary, separated by cases. Currently we assume that there are no repeats in the case_list. Will also perform necessary checks on the
    validity of the task configuration for the given data.
    
    '''
    #First things first, checking that the file extension of the files are even supported! Hacky solution, will require modification at some
    #point in order to cope with more streamlined methods of introducing panoptic or multi-annotator data.
    if ds_configs['file_ext'] not in ['.nii.gz']:
        NotImplementedError('Currently we have not formulated a mechanism for dealing with non nifti filetypes.')

    #Here we will extract some more specific information for the more complex configurations of data, e.g. multi-channel images, 
    # multi-annotator/annotator specific setups, the selection of specific segmentation instances if at all, etc. 
    
    #Here we are extracting the dictionary which contains descriptors about the image data which can be required for this given task.
    #I.e., the description of which image channels to keep under consideration.
    image_conf_dict = extractor(exp_task_configs, ('data_sampling', 'image_conf'))
    annotation_conf_dict = extractor(exp_task_configs, ('data_sampling', 'annotation_conf'))

    # Temporary hack: Checking that all values in image_conf_dict and annotator_conf_dict are lists of length 1
    # Currently we will assume that all of the parameters (values) should be lists of length 1, as we do not yet support cases like 
    # multi-channel, multi-annotator, multi-instance (out-of-the-loop identified) setups. 

    for conf_name, conf_dict in [('image_conf_dict', image_conf_dict), ('annotation_conf_dict', annotation_conf_dict)]:
        for k, v in conf_dict.items():
            if not isinstance(v, list) or len(v) != 1:
                raise ValueError(f"All values in {conf_name} must be lists of length 1, but key '{k}' has value: {v}")
    #Given that multi-instance set-ups will also intrinsically present differing quantities of instances per case, it is challenging to
    #explicitly describe an upper limit on the quantity of instances, like we could potentially for the image channels or #of annotators. 
    
    #Nevertheless, we will also explicitly enforce that we cannot handle an "all" descriptor, for now. 
            if conf_name == 'annotation_conf_dict':
                if "all" in v:
                    raise ValueError('We currently do not support any mechanisms for "all" to be used in the description of the annotation \n' \
                    'strata being used in the task.')
    #Now we load in the full set of dictionaries from the dataset.json for the given "high-level split", and then pick out the
    #the cases in the case list to construct the case_dict.
    default_case_dict = {case:ds_configs[sampling_metadata['split']][case] for case in case_list}
    

    #Now we extract the relevant fields which we want according to the image and annotator configurations.

    #We also put temporary checks to raise any flags for cases of missingness of data with respect to the conf dictionaries
    # for each of the cases. I.e., we do not yet currently support cases where #of channels, instance ids (dummy: 1 for semantic seg), 
    # or the number of annotators are not uniform across the dataset. 
    

    #PLEASE FORGIVE ME FOR THIS MESSY ENTANGLED RECURSIVE CODE. 
    case_list = []
    for case_name, full_case_dict in default_case_dict.items():
        subdict = {'case_name':case_name}
        #Monai dataset constructor requires the use of lists, so we must put the case name as a field in the structure.

        #first we extract the image data, we will raise a flag if any channels in the task config are missing, as we will for now assume 
        #that the number of channels are consistent across all cases and contain a corresponding reference file.
        
        #We use an explicit for-loop for readability here, likely needs to be updated in the future.:
        im_keys = []

        for channel in image_conf_dict['image_channel']:
            if not channel in full_case_dict['images'].keys():
                raise KeyError('Attempted to extract a filepath for an image channel which does not even exist in the reference manifest \n' \
                f'for case {case_name} we currently do not support any missingness in the image channels provided, we currently assume uniformity across all samples in an experiment')
            elif not full_case_dict['images'][channel].endswith('.nii.gz'):
                #Hacky check, will need to modify when we consolidate the utils in the dataset conversion scripts which have some filetype
                #checking.
                raise Exception('The filepath which would be read was from a filetype which is not supported (only nii.gz for now)')
            else:
                im_key = f'image_{channel}'
                subdict[im_key] = create_filepath(dataset_dir=dataset_dir, relpath=full_case_dict['images'][channel])
                if not im_key in im_keys:
                    im_keys.append(f'image_{channel}')
                #TODO: Will need to amend this to potentially handle cases of missingness in the future.
                #We use this method instead of a set because the set introduces randomisation to the ordering, and we do not
                #yet have an appropriate mechanism for preventing this in the dataloader, YET.
        

        #We now do the same for the annotations, a bit more involved, we will retain the hierarchical structure:
        # -annotator
        # |---- semantic_class
        #           |------------instance id. 
        #We will recursively iterate through each of the items in a hardcoded manner... for now we will presume the structure outlined above.
        
        annotation_keys = [] #NOTE: This implementation for tracking keys is extremely hacky and dependent on uniformity across
        #the dataset for the specified task!!!
        
        if not isinstance(annotation_conf_dict['annotator'], list):
            raise TypeError('Current implementation requires all fields of task conf dicts to be lists.')
        for annotator_id in annotation_conf_dict['annotator']:
            
            if annotator_id not in full_case_dict['labels'].keys():
                raise KeyError('Attempted to extract a filepath for annotator which does not exist in the reference manifest \n' \
                f'for case {case_name}, we currently do not support any missingness in annotator ids provided, \n' \
                'we currently assume uniformity across all samples in an experiment')
            else:
                #extract the subdict for the given annotator.
                annotator_subdict = full_case_dict['labels'][annotator_id]
                #now iterate through the semantic classes to check for their presence, requires that all of the semantic classes be present.
                for semantic_class in ds_configs['semantic_classes'].keys():
                    if semantic_class not in annotator_subdict:
                        raise KeyError(f'Missing default semantic class for annotator {annotator_id} in case {case_name}')
                    else:
                        #now we iterate through the given semantic class.... almost there!!
                        sem_class_subdict = annotator_subdict[semantic_class] 
                        if not annotation_conf_dict['instance_id'] == ['instance_1']:
                            raise NotImplementedError('Careful, you have not yet disentangled this complete mess! Still have not provided handling for reading multi-instance')
                        for instance_id in annotation_conf_dict['instance_id']:
                            if instance_id not in sem_class_subdict.keys():
                                raise KeyError('Attempted to extract a filepath which does not exist in the reference manifest \n' \
                                f'for case {case_name}, for semantic class {semantic_class}, for instance id {instance_id} \n' \
                                'we currently assume uniformity across all samples in an experiment') 
                            else:
                                #We will use a very hacky implementation for now, we will designate a very very long string:
                                dict_subpath = ('labels', annotator_id, semantic_class, instance_id)
                                annotation_key = f'{annotator_id}_semclass{semantic_class}_{instance_id}' 
                                subdict[annotation_key] = create_filepath(dataset_dir=dataset_dir, relpath=extractor(full_case_dict, dict_subpath))
                                #NOTE: THIS WONT WORK FOR MULTI-INSTANCE IT IS A TEMPORARY HACK UNTIL I HAVE MY OWN CUSTOM DATALOADERS IMPLEMENTED FOR THIS HIERARCHICAL STRUCTURE.
                                if annotation_key not in annotation_keys:
                                    annotation_keys.append(annotation_key)
                                    #We don't use sets because they have a randomisation component and we want to ensure that 
                                    #order is preserved for now until we have our own custom dataloader.
        case_list.append(subdict) 

    #NOTE: We need a better way of structuring this information hierarchically for the downstream dataloader in the future!
    #For now we can get away with it because we aren't working with multi-annotator or explicitly multi-instance data!!! 
    #(i.e. )
    return case_list, im_keys, annotation_keys 



# def debug_meta(data):
#     print("Image meta:", data['image_T2w'].meta)
#     print("Label meta:", data['annotator_1_semclassbackground_instance_1'].meta)
#     return data

def dataloader_generator(case_list:list, image_keys:list, label_keys:list, transforms_configs:dict):
    '''
    This function handles the construction of a dataset object for iterating through.
    '''
    #Just the basic load transforms in order to load the files in for our custom transforms.
    if monai_version == '1.4.0':
        load_transforms = [
            LoadImaged(keys=image_keys+label_keys, reader="ITKReader", image_only=True),
            EnsureChannelFirstd(keys=image_keys+label_keys),
            Orientationd(keys=image_keys+label_keys, axcodes='RAS'),
            EnsureTyped(keys=image_keys+label_keys, dtype=[torch.float32]*len(image_keys)+[torch.uint8]*len(label_keys)),
        ]
    elif monai_version == '0.9.0':
        load_transforms = [    
            LoadImaged(keys=image_keys+label_keys, image_only=False),
            EnsureChannelFirstd(keys=image_keys+label_keys),
            Orientationd(keys=image_keys+label_keys, axcodes='RAS'),
            MetaTensorConstructor(keys=image_keys+label_keys, dtypes=[torch.float32]*len(image_keys)+[torch.uint8]*len(label_keys)),
        ]
    else:
        raise Exception('Unknown monai version.')
    
    
    #The basic transforms required for all tasks (at least with our current implementation: im_channel merging and sem. seg merging)
    basic_transforms = [
        #Hardcoding in some of these variables for now.
        MergeImChannels(keys=image_keys, channel_list=transforms_configs['image_struct_mapping']['image_channel'] , output_key='image'),
        MergeSegmentations(
            keys=label_keys, 
            sem_mapping=transforms_configs['semantic_class_mapping'], 
            output_sem_code=transforms_configs['config_labels_dict'], 
            output_key='label'),
    ]

    #Hardcoding in the misc transforms for now.. TODO: Add abstraction here!!
    if transforms_configs['non_standard_transfs']:
        if transforms_configs['non_standard_transfs'] == {'component_extraction':'cc_largest'}:
            additional_transforms = [
                KeepTopCC(
                    keys=['label'], 
                    operated_classes=[k for k in transforms_configs['config_labels_dict'] if k.capitalize() != 'Background'],
                    class_code_map=transforms_configs['config_labels_dict'], 
                    component_descriptor='cc_largest',
                    connectivity=3
                    )
            ]
        else:
            raise Exception('There is a non-standard transform that is not supported. Only cc_largest is currently supported.')
    else:
        additional_transforms = []
    all_transforms = load_transforms + basic_transforms + additional_transforms 
    dataset = Dataset(case_list, Compose(all_transforms))    #Compose(load_transforms))
    # return DataLoader(dataset=dataset, batch_size=1, num_workers=4)
 
    return dataset  #Until we can fix the dataloader just use the dataset.. will be slower.


#We will temporarily put some of the custom transforms up here, but it will need to be reshuffled at a later point.
class MergeImChannels:
    #Transform which merges image channels together into a single image instance. Implicitly assumes that the channel_list is the
    # same order as the sequence of the channel list strings. This corresponds to the config which is printed by the exp and also
    # fed into the downstream application. 
    def __init__(self, keys:Sequence, channel_list:list[str], output_key:str ='image'):
        self.input_keys = keys #the set of keys which designate the corresponding locations of the image arrays which are being fused.
        self.output_key = output_key #the key which the merged image will be stored to for downstream extraction. defaults to "image".
        self.channel_list = channel_list #the list of channels which we will be merging.  

        if not all([f'image_{channel}' in keys for channel in self.channel_list]):
            raise Exception('We currently assume that missingness is not permitted, will require special treatment. There was the absence of a required image channel!')

    def merge_channels(self):
        #Method which actally merges the channels together in a MetaTensor construction. Not required for now as we will
        #assume that the tasks are single-channel temporarily.
        raise NotImplementedError('Not implemented yet! Needs to be capable of handling any relevant fields for the ' \
        'multi-channel image data give that we have read-in single channel images.')

    def __call__(self, data):
        d = dict(data)
        #Very hacky solution for now, this is not at all MONAI-like, in the sense that it is not going to iterate over
        #the keys as one typically would approach these types of implementations.
        if len(self.input_keys) == 1:
            print('Single channel image data, just need to re-name the data-struct.')
            d[self.output_key] = d[self.input_keys[0]]   #Removing the deepcopy here... #copy.deepcopy(d[self.input_keys[0]]) 
            #We will handle any stripping of the 
            #irrelevant or non-permitted meta-information downstream. 
        else:
            raise NotImplementedError('Attempted to use a configuration for handling multi-channel image data, which we have not yet configured.')

        return d 
    
class MergeSegmentations:
    #A temporary hack, which will assume single annotator, single instance (or instance as a dummy proxy for semantic seg 
    # datasets)

    #Not been hyperoptimised for efficiency yet as I am sleep deprived and just putting something that is readable for my sleep
    #deprived brain.
    def __init__(self, keys:Sequence, sem_mapping: dict[str, list[str]], output_sem_code: dict[str, int], output_key:str = 'label'):
        self.input_keys = keys #input keys which contain all of the relevant segmentations which will need to be fused.
        self.semantic_map_dict = sem_mapping #The strings which designated the semantic mapps/merging
        if len(self.semantic_map_dict) != 2:
            raise Exception('Hardcoded for now that it is only being applied to binary semantic seg, lets add as many sanity-checks along the way.')        
        self.output_sem_code_dict = output_sem_code #The integer codes which designate what the values
        #are after merging for each of the output-semantic classes.
        
        if self.semantic_map_dict.keys() != self.output_sem_code_dict.keys():
            raise Exception('There was a discrepancy in the task-domain semantic class descriptors for the semantic-mapping dict,' \
            'and the description of the integer-codes that will be allocated for each task-domain semantic class.')
         
        #This is a temporary hack where we presume only semantic labelling is being applied, for now. 
        # Presumes no complexity is going to arise from handling multi-annotator setups or samples with multi-instance explicitly
        # outlined in the input dataset.  
        self.output_key = output_key 
    
    def merge_sem_seg(
            self, 
            seg_array_shape: tuple,
            input_semantic_arrays: dict[str, np.ndarray]):
            #We currently will assume for simplicity that all arrays have been pre-mapped into numpy arrays.
            #TODO: Modify this func and accelerate. 
        '''
        #function which takes the input keys describing the arrays in the  data dictionary, and a description of how the
        # semantic classes need to be mapped in order to generate a semantic segmentation for a basic single-annotator domain.
        '''

        if not isinstance(seg_array_shape, tuple):
            raise TypeError
        else:
            merged_seg_array = np.zeros(seg_array_shape)

        #we create a mapping between the keys describing the arrays, and how we want to merge them.
        #we will be hard-coding with the assumption that we are working with single annotator and single_instance
        #key descriptors for now.
        #TODO: Generalise this function for panoptic & multi-annotator.

        new_sem_mapping = dict() 
        for output_sem, input_sems_list in self.semantic_map_dict.items():
            new_sem_mapping[output_sem] = [f'annotator_1_semclass{sem}_instance_1' for sem in input_sems_list]

        # Check for repeats across all lists. We need more modularity and unit-testing in order to scrap some of these
        # checks! This is just as a precaution!! Cannot have one semantic class be mapped to multiple in the output.

        all_items = [item for sublist in new_sem_mapping.values() for item in sublist]
        if len(all_items) != len(set(all_items)):
            raise ValueError("Duplicate entries found across all lists in new_sem_mapping.")
        #we also check that all of the input/default semantic classes in the dataset have been accounted for, although
        #this has probably been done before my brain cannot retain anything for longer than 10 minutes..we check it by proxy
        #by examining the paths.

        #TODO: This check does not make many assumptions about the source, number of instances etc... but was initially conceived
        #for the single-annotator, single-"instance" semantic seg formulation.
        if not all([input_sem_key in all_items for input_sem_key in self.input_keys]):
            raise Exception('Unaccounted for initial semantic class in the description of the task domain seg task.')
        #Just checking that all of the output semantic keys are available.
        if not all([output_sem_key in self.output_sem_code_dict.keys() for output_sem_key in self.output_sem_code_dict.keys()]):
            raise Exception('Unaccounted for output semantic classes.')
        
        #Now we iterate through and merge binary mask arrays within a task-domain semantic class together, and apply
        #the corresponding integer code which is expected for this.....

        #We recall that it comes in CHWD-like shape (more specifically 1HWD).
        stored_sems = dict() 
        for output_sem_key, sem_pointer_keys_list in new_sem_mapping.items():
            arrays_to_merge = [input_semantic_arrays[sem_key] for sem_key in sem_pointer_keys_list]
            #we check that there are no overlaps here, there must not be because we will be assuming that each voxel is described
            #uniquely by a semantic class! 
            if np.any(np.sum([mask > 0 for mask in arrays_to_merge], axis=0) > 1): 
                #If any voxel has val > 1 then there was an overlap which shouldn't be possible.
                raise ValueError("Overlap detected between diff input semantic class masks, cannot merge together. Re-check.")
            else:
                #Else, we just sum the "foregrounds"
                semclass_arr = np.sum([mask > 0 for mask in arrays_to_merge], axis=0)
                #we store a binary mask and then apply the integer code later.
                stored_sems[output_sem_key] = semclass_arr
        
        #Now we will go through and ensure that there is zero overlap in the foregrounds.
        # Ensure there is zero overlap in the foregrounds (no voxel assigned to more than one class)
        if np.any(np.sum([mask > 0 for mask in stored_sems.values()], axis=0) > 1): 
            #This converts each mask to a binary, then sums over all masks. If any voxel has val > 1 then there was an overlap.
            raise ValueError("Overlap detected between semantic class masks in the merging operation.")
        else:   
            merged_seg_array[0, ...] = np.sum([stored_sems[class_lb] * class_code for class_lb, class_code in self.output_sem_code_dict.items()], axis=0)

        
        if merged_seg_array.sum() == 0:
            warnings.warn('Beware, empty annotation! Please check if this is intended.')
        return merged_seg_array 
    
    def __call__(self, data):
        d = dict(data)
        copied_metatensor_template = copy.deepcopy(d[self.input_keys[0]])
        #We check that the metatensor_template is actually a metatensor.
        if not isinstance(copied_metatensor_template, MetaTensor):
            raise TypeError('The metatensor template for storing the merged segmentation merging was not a metatensor')
        #TODO: Re-check whether the metadictionary or stored transforms will introduce problems downstream
        #due to the memory of transforms applied............
        
        #Our hand is temporarily forced to use such a hacky method for generating a MetaTensor because we require the 
        #trace of the transform/transform history! Same reason we also didn't inherit from the base Transform class so that
        #this transform could remain hidden.
        
        #TODO: Find a better resolution to this hack.
        if monai_version == '1.4.0':
            # input_semantic_arrays = copy.deepcopy({k:d[k].data for k in self.input_keys}) #copy.deepcopy({k:d[k].array for k in self.input_keys})
            input_semantic_arrays = {k:d[k].data for k in self.input_keys}
        elif monai_version == '0.9.0':
            # input_semantic_arrays = copy.deepcopy({k:d[k].data for k in self.input_keys})
            input_semantic_arrays = {k:d[k].data for k in self.input_keys}
        else:
            raise Exception('Unknown monai version!')

        #We check that the datatype of the structure is all uniform. Ulimately the datastructure is implicitly contained.
        dtypes = [arr.dtype for arr in input_semantic_arrays.values()]
        if len(set(dtypes)) > 1:
            raise TypeError(f"Not all arrays in input_semantic_arrays have the same instance type: {set(dtypes)}")
        
        #We assign what the output data structure should be like for storing. 
        if isinstance(dtypes[0], torch.dtype):
            output_type = torch.Tensor
        elif isinstance(dtypes[0], np.dtype):
            output_type = np.ndarray 
        else:
            raise TypeError('Unknown datatype')
        output_dtype = dtypes[0] 

        #We now map all the arrays into numpy arrays for simplicity.
        if output_type == torch.Tensor:
            input_semantic_arrays = {k:v.cpu().numpy() for k, v in input_semantic_arrays.items()}
        elif output_type == np.ndarray:
            input_semantic_arrays = {k:np.array(v, copy=False) for k, v in input_semantic_arrays.items()} 
        else:
            raise Exception('Unsupported datatype obtained after the .data operation')

        array_shapes = [arr.shape for arr in input_semantic_arrays.values()]
        #hardcoding a check that it must be single-channel, 4D volume. TEMPORARY. We just reuse the array shapes.
        if len(set(array_shapes)) > 1:
            raise ValueError('There was an inconsistency in the shape of the arrays')
        if any([len(array_shape) != 4 or array_shape[0] != 1 for array_shape in array_shapes]):
            raise ValueError('There was an array which was either not single-channel or not 4D')
        
        output_shape = array_shapes[0] 
        

        merged_seg = self.merge_sem_seg(
            seg_array_shape=output_shape,
            input_semantic_arrays=input_semantic_arrays)

        # assert isinstance(merged_seg, np.ndarray), 'For now we have written the transform to require handling of np arrays'
        assert merged_seg.shape == output_shape, 'The merged seg did not have the correct shape as required for a 1HWD semantic seg.'

        #Mapping the seg back into the expected datastructure.

        # Preserve both type and dtype. For now the outputted merg_seg should always be a numpy array but we will
        # just be careful anyways.
        if output_type == np.ndarray:
            if not isinstance(merged_seg, np.ndarray):
                merged_seg = np.array(merged_seg)
            merged_seg = merged_seg.astype(output_dtype, copy=False)
        elif output_type == torch.Tensor:
            if not isinstance(merged_seg, torch.Tensor):
                merged_seg = torch.from_numpy(merged_seg)
            merged_seg = merged_seg.to(output_dtype)
        else:
            raise TypeError('The output array datatype was unknown')
        
        #Placing back into the metatensor datastructure.
        #TODO: Fix this when your brain is actually working again. You're pretty much moving between datatypes for
        #no reason.
        if monai_version == '1.4.0':
            if isinstance(merged_seg, torch.Tensor):        
                copied_metatensor_template.array = merged_seg.numpy() 
            elif isinstance(merged_seg, np.ndarray):
                copied_metatensor_template.array = merged_seg
            else:
                raise Exception 
        elif monai_version == '0.9.0':
            if isinstance(merged_seg, torch.Tensor):
                copied_metatensor_template.data = merged_seg 
            elif isinstance(merged_seg, np.ndarray):
                copied_metatensor_template.data = torch.from_numpy(merged_seg)
            else:
                raise Exception 
        d[self.output_key] = copied_metatensor_template 

        return d  

class KeepTopCC: 
    #This transform is actually implemented in MONAI, but temporarily we will write our own because we have not yet consolidated 
    # the different branches into one. For now we elect not to make use of class inheritance wrt wrapping the base Transform classes
    # until the branches are merged. 

    #Only keeps the largest region, not the N largest regions. 
    def __init__(self, 
                keys, 
                operated_classes:list[str], 
                class_code_map:dict[str, int],
                component_descriptor = str,
                connectivity: int = None):
        self.keys = keys #Just a temporary hacky method. We implicitly assume that this transform will occur in-place.
        
        warnings.warn('This transform is only intended, currently, for cases where num_components (connectivity-wise) should be low'
        'although the connectivity function has been provided ample flexibility to prevent overflows (int32). ' \
        'The downstream retained components are assumed to be sufficiently low in number to be able to be stored in a uint8 array. \n' \
        'This is a temporary hack until we have a more robust implementation of the connected component analysis transform.'
        'Very possible that this can cause memory issues even with int32.......')
        
        if component_descriptor != 'cc_largest':
            raise NotImplementedError('We have only formulated this transform for the purpose of finding the largest cc of each semantic class.')
        
        if connectivity is None:
            self.connectivity = None 
            print('Performing n_dim connectivity for n_dim arrays.')
        else:
            self.connectivity = connectivity
        #The number of orthogonal jumps permitted. Should by default set this to n_dim, which it will for None. 

        self.operated_classes = operated_classes #This designates the classes for which this operation will be applied.
        if 'background' in self.operated_classes or 'Background' in self.operated_classes:
            raise Exception('Should not be operating CC analysis on the background semantic classes as it has 0 meaning for objectness!')
        self.class_code_map = class_code_map 
        #This is a dictionary which will denote the numeric representation for the classes
        #which are being operated upon for the connected component analysis! 

    def get_largest_cc(self, mask):
        #As of now, this function is only designed with semantic segmentation in mind. We are assuming that voxels are uniquely
        #assigned. Although this is unlikely to change for panoptic segmentation, it is worthy to note for when I have less
        #weary eyes.
        largest_cc = np.zeros(shape=mask.shape, dtype=mask.dtype)
        if not largest_cc.ndim == 4:
            raise Exception('Currently we are only working in the domain of volumetric segmentation.')
        if not largest_cc.shape[0] == 1:
            raise Exception('Currently we assume only semantic segmentation implementation, and so we require that the input seg be single channel!')
        #We must be aware that the image array will be 4-dimensional (1HWD structure) and so we must account for that when
        #extracting the connected component masks! 
        analysed_mask = mask[0] #extracting the actual spatial component only.

        #for simplicitly, we will create a set of arrays by semantic class, and then fuse them afterwards.
        stored_ccs = dict()
        for class_lb, class_code in self.class_code_map.items():
            if class_lb.capitalize() == 'Background':
                if class_code != 0:
                    raise ValueError('We assert that the background semantic class will always be assigned to a voxel value of 0.' \
                    'Another potentially pointless check in case my sleep-deprived brain has not recalled' \
                    'I put this check elsewhere already.')
                #Meaningless to perform cc analysis on the background semantic class. Just return a binary mask!
                #NOTE: Implicitly assumes that the semantic code for background is always going to be 0 for this implementation!
                stored_ccs[class_lb] = np.zeros(analysed_mask.shape).astype(dtype=mask.dtype)
            if class_lb not in self.operated_classes:
                #We will only be performing on the classes which are designated as being operated upon.
                #in this case just return the binary mask!
                stored_ccs[class_lb] = np.where(analysed_mask == class_code, 1, 0).astype(dtype=mask.dtype) 
            else:
                fg_mask = np.where(analysed_mask == class_code, 1, 0).astype(dtype=np.int32) 
                cc_analysed_mask, num_cc = cc_label(label_image=fg_mask, background=0, return_num=True, connectivity=self.connectivity)
                if num_cc == 0 or cc_analysed_mask.sum() == 0:
                    warnings.warn(f'Pre-warning, empty fg in the semantic class {class_lb}')
                    stored_ccs[class_lb] = np.zeros(analysed_mask.shape).astype(dtype=mask.dtype)
                    # We return an empty array.
                else:
                    largest_region_code = np.argmax(np.bincount(cc_analysed_mask.flat)[1:]) + 1 # + 1 because we indexed out the bg, 
                    #but np.argmax will return values 0 indexed, which could give us largest_region_code = 0 despite this being bg!!
                    if not largest_region_code > 0 or not largest_region_code <= num_cc:
                        raise ValueError('Somehow we managed to get an invalid number of cc under the subloop which handles cases which were not empty')
                    stored_ccs[class_lb] = np.where(cc_analysed_mask == largest_region_code, 1, 0).astype(dtype=mask.dtype)
                    # warnings.warn(f'Not really a warning, but voxel count in remaining component for semantic class {class_lb} had {stored_ccs[class_lb].sum()} voxels')
        #Now we will go through and ensure that there is zero overlap in the foregrounds.
        # Ensure there is zero overlap in the foregrounds (no voxel assigned to more than one class)
        if np.any(np.sum([mask > 0 for mask in stored_ccs.values()], axis=0) > 1): 
            #This converts each mask to a binary, then sums over all masks. If any voxel has val > 1 then there was an overlap.
            raise ValueError("Overlap detected between semantic class masks in connected component outputs.")
        elif np.any([np.where(mask_check == self.class_code_map[class_lb],1,0).sum() > np.where(analysed_mask == self.class_code_map[class_lb], 1, 0).sum() for class_lb, mask_check in stored_ccs.items()]):
            #This checks that none of the stored ccs have more voxels than the original fg for each semantic class.
            raise ValueError("At least one of the semantic classes has more voxels in the retained connected components than the original foreground mask. \n" \
            'This is likely due to a bug in the connected component analysis implementation, please re-check.')
        else:
            largest_cc[0, ...] = copy.deepcopy(np.sum([stored_ccs[class_lb] * class_code for class_lb, class_code in self.class_code_map.items()], axis=0))
            #We deepcopy here because it should allow us to delete the other variables without references being kept so that
            # the memory can be freed up.        
        del fg_mask
        del analysed_mask
        del stored_ccs
        gc.collect() #We will try to free up some memory here, although this is not guaranteed to work as there are references

        return largest_cc

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            
            if monai_version == '1.4.0':
                orig_array = copy.deepcopy(d[key].data)
            elif monai_version == '0.9.0':
                orig_array = copy.deepcopy(d[key].data)
            else:
                raise Exception('Unknown monai version!')

            output_dtype = orig_array.dtype
            #We assign what the output data structure should be like for storing. 
            if isinstance(output_dtype, torch.dtype):
                output_type = torch.Tensor
            elif isinstance(output_dtype, np.dtype):
                output_type = np.ndarray             
            else:
                raise Exception('Unknown datastructure')

            if output_type == torch.Tensor:
                numpified_array = orig_array.cpu().numpy()
            elif output_type == np.ndarray:
                numpified_array = np.array(orig_array, copy=False)
            else:
                raise Exception('Unsupported datatype')
        
            result = self.get_largest_cc(numpified_array)

            # Preserve both type and dtype. For now the output should always be a numpy array but we will
            # just be careful anyways.
            if output_type == np.ndarray:
                if not isinstance(result, np.ndarray):
                    result = np.array(result)
                result = result.astype(output_dtype, copy=False)
            elif output_type == torch.Tensor:
                if not isinstance(result, torch.Tensor):
                    result = torch.from_numpy(result)
                result = result.to(dtype=output_dtype)
            else:
                raise TypeError('Unknown datastructure')
            
            assert result.shape[0] == 1, 'We currently only assume this implementation is meant to semantic segmentation, will require expansion.'
        
            #Placing back into the metatensor datastructure.
            if monai_version == '1.4.0':
                if isinstance(result, torch.Tensor):  #Just in case. Should be a numpy array however.    
                    d[key].array = result.numpy() 
                elif isinstance(result, np.ndarray):
                    d[key].array = result
                else:
                    raise Exception 
            elif monai_version == '0.9.0':
                if isinstance(result, torch.Tensor):
                    d[key].data = result  
                elif isinstance(result, np.ndarray):
                    d[key].data = torch.from_numpy(result)
                else:
                    raise Exception 

        del numpified_array
        del orig_array 
        gc.collect() 

        return d



def iterate_dataloader_check(data_instance):
    
    if not isinstance(data_instance, dict):
        raise TypeError('Data loader requires dictionary based transforms to be passed through the dataloader')
            

    if isinstance(data_instance['image'], MetaTensor) and isinstance(data_instance['label'], MetaTensor):
        try:
            im_meta_dict = data_instance['image'].meta #['image_meta_dict']
            label_meta_dict = data_instance['label'].meta #['label_meta_dict']
        except:
            raise Exception('The loaded data instance does not contain a meta dictionary')


        #TODO: Put back in some checks on the metadata potentially... although i am going to wipe almost all of it anyways.
        #TODO: Need to expand this to be flexible to multi-instance and multi-annotator setups. 

        #We assert that the data must be single channel for our current application!
        # if int(im_meta_dict['pixdim[4]']) != 1:
        if data_instance['image'].shape[0] != 1: 
            raise ValueError('This application only currently supports single channel implementations.')
        # if int(label_meta_dict['pixdim[4]']) != 1:
        if data_instance['label'].shape[0] != 1:
            raise ValueError('This application only currently supports semantic segmentation implementations')
    else:
        raise TypeError("The loaded image and label data must be a MetaTensor")

    



def data_instance_reformat(data_instance:dict):
     
    '''
    This function reformats the data instance from the output of the dataset generator's load transforms into the 
    data_instance reformat expected by the pseudo-ui.
    '''
    if not isinstance(data_instance, dict) or not data_instance:
        raise Exception('Data instance is assumed to be a non-empty dictionary format..')

    #We extract the paths from the meta info: Deprecated.

    # im_path = copy.deepcopy(data_instance['image'].meta['filename_or_obj'])
    # label_path = copy.deepcopy(data_instance['label'].meta['filename_or_obj'])

    # if not os.path.exists(im_path) or not os.path.exists(label_path):
    #     raise Exception('One of the paths does not exist! Or the full absolute path was not provided to the dataset constructor')
    
    #We will deepcopy like a paranoid person for now. 

    #Actually deepcopying is going to kill the memory.
    case_name = copy.deepcopy(data_instance['case_name'])

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
    #NOTE: We will NOT be copying actually.
    im_meta_dict = im_tensor.meta #copy.deepcopy(im_tensor.meta)
    label_meta_dict = label_tensor.meta #copy.deepcopy(label_tensor.meta)

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
    #Probably should have already been altered because the reference is the same, but just to be sure.
    im_tensor.meta = im_meta_dict #copy.deepcopy(im_meta_dict)
    label_tensor.meta = label_meta_dict #copy.deepcopy(label_meta_dict)
    


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
    
    # if os.path.split(im_path)[1].split('.')[0] != os.path.split(label_path)[1].split('.')[0]:
    #     raise Exception('The name for the provided image and label needs to be the same!')
    

    # patient_name = os.path.split(im_path)[1].split('.')[0]

    #HACK:
    del data_instance 
    gc.collect() #Garbage collection to free up memory, this is a hacky solution for now.
    torch.cuda.empty_cache() #Emptying the cuda cache... unlikely that this would actually be required. 

    return reformat_data_instance, case_name #patient_name


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
        

class MetaTensorConstructor: 
    #Very old and probably not robust for newer versions of monai, thankfully we won't need it for that anyways at a later point.
    def __init__(self, keys, dtypes):
        self.keys = keys
        self.dtypes = dtypes 
    def __call__(self, data):
        if not monai_version == '0.9.0':
            raise Exception('Constructor was only implemented as a temporary stand-in.')


        d = dict(data)
        for idx, key in enumerate(self.keys):
            im = d[key] 
            meta_dict = d[f'{key}_meta_dict']
        
            modif_im = torch.from_numpy(im).to(dtype=self.dtypes[idx])#torch.float32)
            modif_meta = {
                'original_affine': torch.from_numpy(meta_dict['original_affine']).to(dtype=torch.float32), 
                'affine': torch.from_numpy(meta_dict['affine']).to(dtype=torch.float32), 
                # 'filename_or_obj': meta_dict['filename_or_obj'],
                # 'pixdim': meta_dict['pixdim']
                }
            
            d[key] = MetaTensor(x=modif_im, meta=modif_meta)
        return d