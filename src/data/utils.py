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
        'semantic_class_mapping':semantic_class_mapping,
        'image_struct_mapping': extractor(exp_task_configs, ('data_sampling', 'image_conf')),
        'annotation_struct_mapping':extractor(exp_task_configs, ('data_sampling', 'annotation_conf'))
    }

    #Now we add any other non-semantic class mapping transforms, i.e. the more atypical ones (with respect to the current
    # common conventions). This is just a temporary hacky fix until we build a proper infrastructure for the dataloading pipeline.
    transforms_configs.update({k:v for k,v in exp_task_configs['data_transforms'].items() if k != 'semantic_class_mapping'})

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





def dataloader_generator(case_list:list, image_keys:list, label_keys:list, transforms_configs:dict):
    '''
    This function handles the construction of a dataset object for iterating through.
    '''
    load_transforms = [
        LoadImaged(keys=['image', 'label'], reader="ITKReader", image_only=True),
        EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        EnsureTyped(keys=['image', 'label'], dtype=[torch.float64, torch.int64]),
    ]
    dataset = Dataset(case_list, Compose(load_transforms))
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