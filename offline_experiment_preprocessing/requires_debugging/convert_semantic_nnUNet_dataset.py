#Equivalent script for reformatting datasets, but this time from the nnU-Net convention into our requirements as the nnU-Net convention is also quite common.
#Borrowing skeleton from convert_MSD_dataset.py, with some modifications for our own requirements. Only intended for re-structuring semantic segmentation
#datasets. 

##########################################################################################

#Our constraints on the nnu-net style are as follows:

# We assume that the imagesTr, labelsTr, imagesTs and labelsTs are all still named similarly to MSD. But we loosen the requirement that imagesTs and labelsTs be 
# required.

# We assume that a dataset.json file is provided which aligns with the nomenclature assumed by nnU-net's convention. 

# We assume that nnU-net style is constrained to semantic segmentation (as this was the original use-case), and so we presume that this still holds.

# We do not assume that the dataset is multi-annotator a priori (instead we assume it is single-annotator/label fused because so so few of them are. It would be 
# quicker to manually perform conversion for those (or to use this script as part of a conversion pipeline).

import argparse
import multiprocessing
import shutil
from typing import Optional
import SimpleITK as sitk
import os 
import sys 
import copy 
import numpy as np
import json
import warnings 
datasets_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),'datasets')
from utils import (
    check_dataset_existence, 
    extract_nifti_files, 
    extract_subfiles, 
    extract_subdirs, 
    load_json, 
    save_json, 
    full_path_splitter, 
    file_ext_splitter,
    # split_semantic_seg_nifti,
    SUPPORTED_DOWNSTREAM_MEDIO_FILE_EXT,
    SUPPORTED_UPSTREAM_MEDIO_FILE_EXT,
    SUPPORTED_UPSTREAM_NONMEDIO_FILE_EXT
)

#Just a list of the input-output pairs which are supported with respect to the filetype.
SUPPORTED_IN_OUT_IO_MAPS = [
    ['.nii.gz', '.nii.gz']
]

def image_filetype_restructure(case:str, case_channels:list, source_folder:str, target_folder:str, file_ext:str):
    '''
    Function which restructures the nnu-net formatted image data, into the structure assumed by our temporary folder structure. Assumption: same file_ext.

    case: The name of the corresponding case for restructuring 
    case_channels: The image channels corresponding to a given "case" or "sample". 
    source_folder: The absolute path to folder which contains the images being copied over.
    target_folder: The outer path which the image cases will be placed within.
    file_ext: Self explanatory.
    '''

    os.makedirs(os.path.join(target_folder, case) ,exist_ok=False)
    for channel in case_channels:
        shutil.copy(os.path.join(source_folder, channel + file_ext), os.path.join(target_folder, case, channel + file_ext))


def io_read(path: str, file_ext: str):
    '''
    Function which reads an input path, according to the file_ext type. We format it like this, just in case we require file type specific operations.
    '''
    if file_ext == '.nii.gz':
        return sitk.ReadImage(path) 
    else:
        raise NotImplementedError('Was provided with a filetype which is not supported.') 

def io_write(img: np.ndarray, metadata: dict, output_folder:str, folder_exist_ok:bool, filename:str, file_ext: str):
    '''
    Function which writes an image, according to the file_ext type.
    
    Relies on the image array (assumed to be a numpy ndarray) and a metadata dictionary, which should contain all the parameters required for the given file ext type.
    '''
    os.makedirs(output_folder, exist_ok=folder_exist_ok)
    if file_ext == '.nii.gz':
        #Requires spacing, origin and orientation.
        required_fields = ['spacing', 'origin', 'direction']
        if not all([field in metadata.keys() for field in required_fields]):
            raise Exception('Missing metadata required for saving the image data')
        else:
            #Make some assertions on the structure of the metadata, because we are working exclusively with 3D volumes.
            assert len(metadata['spacing']) == 3
            assert len(metadata['origin']) == 3
            assert len(metadata['direction']) == 9

            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(metadata['spacing'])
            img_itk_new.SetOrigin(metadata['origin'])
            img_itk_new.SetDirection(metadata['direction'])
            sitk.WriteImage(img_itk_new, os.path.join(output_folder, filename + file_ext))
    else:
        raise NotImplementedError('Was provided with a filetype which is not supported')


def image_filetype_conversion(in_out_file_ext: list[str]):

    if in_out_file_ext not in SUPPORTED_IN_OUT_IO_MAPS:
        raise Exception('The selected conversion was between filetypes which do not have a supported filetype conversion')

    if in_out_file_ext[0] == in_out_file_ext[1]:
        warnings.warn('This function is intended for converting between image filetypes, it is unnecessary for nnu-net style formatted datasets. Skipping...')

def sem_seg_split(
        source_folder:str, 
        target_folder:str, 
        case:str, 
        in_out_file_ext:list[str], 
        semantic_class_dict: dict):
    '''
    Function which splits the segmentations into the folder structure (and filetype) that is expected. Written slightly more abstractly, 
    in case distinctions are required in the future, just as a starting point.
    Although... I assume very little distinction would be required across some of the more popular medIO file_ext types (e.g., nifti, dicom, etc.)

    args:
    source_folder: Path to the source folder for the annotations.
    target_folder: Path to the folder for storing all of the class-separated annotations
    case: String denoting the sample/case.
    in_out_file_ext: A pair, denoting the input file_type and the output file_type
    semantic_class_dict: A dictionary containing the semantic class (words): integer code representations (in str or int)

    '''
    if in_out_file_ext not in SUPPORTED_IN_OUT_IO_MAPS:
        raise Exception('The selected conversion was between filetypes which do not have a supported filetype conversion')

    if in_out_file_ext[0] == in_out_file_ext[1]:
        #When the in-out maps are the same filetypes, considerably less headache.
        if in_out_file_ext[0] == '.nii.gz': #Reusing most of the code written for the MSD conversion script.
            sitk_orig_seg = io_read(os.path.join(source_folder, case + in_out_file_ext[0]), in_out_file_ext[0])
            dim = sitk_orig_seg.GetDimension()
            #We presume only 3D volumes for the segmentations! Purely spatial-component assumption (4D volumes for images are assumed to be co-registered spatially)
            if dim != 3:
                raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, case))
            else:
                seg_npy = sitk.GetArrayFromImage(sitk_orig_seg)
                metadata = {
                    'spacing':sitk_orig_seg.GetSpacing(),
                    'origin':sitk_orig_seg.GetOrigin(),
                    'direction':sitk_orig_seg.GetDirection()
                }
                #We partition according to semantic class codes in the semantic_class_dict. Within each we remap to a "single instance", we do not assume distinct id's 
                #for "stuff" classes but will use this same convention for consistency.
                for word, code in semantic_class_dict.items():
                    if isinstance(code, int):
                        seg_class_word = np.where(seg_npy == code, 1, 0)
                    elif isinstance(code, str): 
                        code = int(code) #Throws an error if the code is a float, it must be an int even if expressed as a str.
                        seg_class_word = np.where(seg_npy == code, 1, 0)
                    #Creating the subfolder for each semantic class for each sample.
                    semantic_class_subfolder = os.path.join(target_folder, f'semantic_class_{word}')
                    io_write(seg_class_word, metadata, semantic_class_subfolder, False, case + "_%04.0d" % 1, in_out_file_ext[1]) 
                    #Given that semantic segmentation is being mapped to a single "instance", we will just hard-code the "instance".
        else:
            raise NotImplementedError('Unsupported conversion was not flagged somehow.')
    else:
        raise NotImplementedError('Unsupported conversion was not flagged somehow.')
    

def convert_nnunet_style_dataset(source_folder: str,
                        dataset_target_id: int,
                        input_file_ext: str = '.nii.gz', #We currently assume a fixed/consistent file_extension across all imaging samples across a dataset.   
                        output_file_ext: str = '.nii.gz', #An output file ext which controls what filetype we desire/require for downstream.  
                        process_Ts: bool = False, 
                        num_processes: int = 1) -> None:
    
    if input_file_ext not in SUPPORTED_UPSTREAM_MEDIO_FILE_EXT + SUPPORTED_UPSTREAM_NONMEDIO_FILE_EXT:
        raise Exception('The script does not contain functionality to read ')
    if output_file_ext not in SUPPORTED_DOWNSTREAM_MEDIO_FILE_EXT:
        raise Exception('The selected file extension for output (used in downstream applications) is not supported, yet.')

    if source_folder.endswith('/') or source_folder.endswith('\\'):
        source_folder = source_folder[:-1] 

    #We assume some bare minimum for the structure of the datasets... 
    labelsTr_path = os.path.join(source_folder, 'labelsTr')
    imagesTr_path = os.path.join(source_folder, 'imagesTr')
    labelsTs_path = os.path.join(source_folder, 'labelsTs')
    imagesTs_path = os.path.join(source_folder, 'imagesTs')
    
    assert os.path.isdir(labelsTr_path) and os.path.exists(labelsTr_path), f"labelsTr missing in source folder or was not a subdirectory."
    assert os.path.isdir(imagesTr_path) and os.path.exists(imagesTr_path), f"imagesTr missing in source folder or was not a subdirectory."
    if process_Ts:
        assert os.path.isdir(labelsTs_path) and os.path.exists(labelsTs_path), "labelsTs missing in source folder or was not a subdirectory."
        assert os.path.isdir(imagesTs_path) and os.path.exists(imagesTs_path), f"imagesTs missing in source folder or was not a subdirectory."

    dataset_json = os.path.join(source_folder, 'dataset.json')
    assert os.path.isfile(dataset_json), f"nnu-net like formatted dataset.json was missing in source_folder"


    # Extract the dataset name, we generally just assume that it is the dataset itself..we will a priori assume that the name of the dataset is fully alphanumeric 
    # with no spaces.
    dataset_name = os.path.basename(source_folder)

    # check if target dataset id is already being used for another dataset
    existing_datasets = check_dataset_existence(datasets_path, dataset_target_id)
    assert len(existing_datasets) == 0, f"Target dataset id {dataset_target_id} is already taken, please consider changing " \
                                        f"it. Conflicting dataset: {existing_datasets}"

    target_dataset_name = f"Dataset{dataset_target_id:03d}_{dataset_name}"
    target_folder = os.path.join(datasets_path, target_dataset_name)
    target_imagesTr = os.path.join(target_folder, 'imagesTr')
    target_imagesTs = os.path.join(target_folder, 'imagesTs')
    target_labelsTr = os.path.join(target_folder, 'labelsTr')
    target_labelsTs = os.path.join(target_folder, 'labelsTs')
    
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)
    if process_Ts:
        os.makedirs(target_imagesTs, exist_ok=True)
        os.makedirs(target_labelsTs, exist_ok=True) 

    #Loading the dataset json, will be reformatted but also used for splitting semantic seg. labels.
    dataset_json = load_json(dataset_json)

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []

        # convert 4d train images. We presume an nnu-net format where the filename (without the file_ext) has an ending which denotes the channel name. Everything else
        #prior should refer to the same "case". 

        #Given the nnu-net style, we presume that the files are named: filename_XXXX + ext where XXXX denotes some channel dependent information. We can hardcode that,
        #it should just pass through the string extracted prior to the final underscore _, as per nnu-net convention. 

        source_images = [file_ext_splitter(i, suffix=input_file_ext)[0] for i in extract_subfiles(imagesTr_path, suffix=input_file_ext, join=False) if
                         not i.startswith('.') and not i.startswith('_')]
        #First we just extract the full list of the files in the folder.

        #Sorting filenames into the different cases, this is done by presuming a fixed prefix for each case, first we search, then we go back through and bin.
        #Given our assumption about the structure about nnu-net style file naming convention:
        cases = list(dict.fromkeys([filename[:filename.rindex('_')] for filename in source_images])) #Instead of using sets which eradicate ordering, we use this strategy.

        #We partition the dataset into the "samples/cases" folders, i.e., whatever the dataset used to stratify a set of images into a 
        # given sample/case. This is with the assumption that these can come as 4D volumes. Other dataset conventions, of course, do not necessarily assume this. 
        cases_split = [[j for j in source_images if j.startswith(i)] for i in cases] 
        #Using this strategy for splitting by case as the number of channels may not be fixed across all cases.
        
        #In the case where there is no require conversion between filetypes.
        if input_file_ext == output_file_ext:
            if output_file_ext in SUPPORTED_DOWNSTREAM_MEDIO_FILE_EXT:
                #In this case, we just need to copy over the image files, but just restructured for our downstream use case. 
                #ASSUMING that it is a medical imaging filetype which contains meta information!
                # image_filetype_restructure(cases[0], cases_split[0], imagesTr_path, target_imagesTr, output_file_ext)
                results.append(
                    p.starmap_async(
                        image_filetype_restructure, zip(cases, cases_split, [imagesTr_path] * len(cases), [target_imagesTr] * len(cases), [output_file_ext] * len(cases))
                    )
                )
            else: 
                TypeError('Downstream implementations presume MedIO filetypes, so it is currently not permitted to just copy over non-MEDIO filetypes, e.g., TIFF. Should have been flagged already.')
        else:
            #In this case, we require some conversion between filetypes
            if input_file_ext in SUPPORTED_UPSTREAM_MEDIO_FILE_EXT:
                # First for the conversion between MEDIO types:
                target_sample_folders = copy.deepcopy(cases) 
                
                raise NotImplementedError
            
            elif input_file_ext in SUPPORTED_UPSTREAM_NONMEDIO_FILE_EXT:
                #Now for conversion from a non-MEDIO type to a MedIO filetype.
                NotImplementedError('Dataset conversion not yet implemented for non MedIO filetypes, e.g., TIFF.')
            else:
                TypeError('The input filetype is not supported at all.')


        # convert "training set" segmentations

        #In this script, we will just assume it is fully semantic segmentation. Not worth performing label remappings for segmentation tasks which are not 
        # fully disjoint in the loop of a dataloader, as it would be preferable to double check the outputs & because we don't have any good strategies for robustly
        # splitting semantic masks into instance or panoptic masks without connected component analysis (which is not good enough for all cases!) 
        #
        # That is an open problem?..

        # We also need to split the segmentations and restructure the segmentations, for downstream use case (with our assumed temporary folder structure).
        target_sample_folders = [os.path.join(target_labelsTr, 'annotator_1', case) for case in cases] #We initially said that this script will assume the single annotation
        #scenario (not a multi-annotator one).
        if input_file_ext == output_file_ext:
            if output_file_ext in SUPPORTED_DOWNSTREAM_MEDIO_FILE_EXT:
                #In this case, we need to split the semantic segmentation into "one-hot" binary masks ("single-instance") for our downstream use case.
                #ASSUMING that it is a medical imaging filetype which contains meta information!
                results.append(
                    p.starmap_async(
                        sem_seg_split, zip([labelsTr_path] * len(cases), target_sample_folders, cases, [[input_file_ext, output_file_ext]] * len(cases), [dataset_json['labels']] * len(cases))
                    )
                ) 

            else: 
                TypeError('Downstream implementations presume MedIO filetypes. Should have been flagged already.')
        else:
            #In this case, we require some conversion between filetypes
            if input_file_ext in SUPPORTED_UPSTREAM_MEDIO_FILE_EXT:
                # First for the conversion between MEDIO types:
                target_sample_folders = copy.deepcopy(cases) 
                
                raise NotImplementedError
            
            elif input_file_ext in SUPPORTED_UPSTREAM_NONMEDIO_FILE_EXT:
                #Now for conversion from a non-MEDIO type to a MedIO filetype.
                NotImplementedError('Dataset conversion not yet implemented for non MedIO filetypes, e.g., TIFF.')
            else:
                TypeError('The input filetype is not supported at all.')
        
        


        #The vast majority of datasets are single-annotator (only one annotation provided/pre-fused), instance_id = 1 for all semantic classes.  
        
        # target_sample_folders = [os.path.join(target_labelsTr, file_ext_splitter(full_path_splitter(i)[-1],suffix='.nii.gz')[0], 'annotator_1') for i in source_images]
        # results.append(
        #     p.starmap_async(
        #         split_semantic_seg_nifti, zip(source_images, target_sample_folders, [dataset_json['labels']] * len(source_images))
        #     )
        # )



        # convert 4d test images
        # source_images = [i for i in extract_nifti_files(imagesTs_path, join=False, sort=True) if
        #                  not i.startswith('.') and not i.startswith('_')]
        # source_images = [os.path.join(imagesTs_path, i) for i in source_images]
        # target_sample_folders = [os.path.join(target_imagesTs, file_ext_splitter(full_path_splitter(i)[-1],suffix='.nii.gz')[0]) for i in source_images]
        # results.append(
        #     p.starmap_async(
        #         split_4d_nifti, zip(source_images, target_sample_folders)
        #     )
        # )

        # #################################################################################################################################################


        # #We add the option (in case the folder exists) to also process LabelsTs. We presume that this folder will have the same structure as LabelsTr in relation to 
        # #imagesTs.
        # if process_Ts:
        #     source_images = [i for i in extract_nifti_files(labelsTs_path, join=False, sort=True) if
        #                  not i.startswith('.') and not i.startswith('_')]
        #     source_images = [os.path.join(labelsTs_path, i) for i in source_images]

        #     target_sample_folders = [os.path.join(target_labelsTs, file_ext_splitter(full_path_splitter(i)[-1],suffix='.nii.gz')[0], 'annotator_1') for i in source_images]
        #     # split_semantic_seg_nifti(source_images[0], target_sample_folders[0], dataset_json['labels'])
        #     results.append(
        #         p.starmap_async(
        #             split_semantic_seg_nifti, zip(source_images, target_sample_folders, [dataset_json['labels']] * len(source_images))
        #         )
        #     )
        
        


        [i.get() for i in results]

    
    #Remapping the dataset configs into an nnU-Net inspired style for the channel naming convention. 
    # Appending the file extension to the set of configs for downstream functions to be able to exploit. 
    # (We will later require some adaptation to nrrd files for instance segmentation/panoptic seg. labels)
    
    #We extend the dataset.json to include information about whether semantic classes will be contained within each sample, and also pre-normalise the integer codes.
    #We always assume background = 0. (These are already done for MSD but good to be consistent just in case.)
    
    semantic_labels_dict = dict()
    for idx, (code, class_lb) in enumerate(dataset_json["labels"].items()):
        if class_lb.title() == "Background":
            if int(code) != 0:
                raise Exception("Unexpected semantic code for background class.")
            semantic_labels_dict[class_lb] = {"id": "0", "optional":True, "semantic_type": "stuff"}
        else:
            semantic_labels_dict[class_lb] = {"id": str(idx), "optional": True, "semantic_type": "stuff"}
    #We re-order by the class_id, always in increasing order starting at 0 (for background), typically this will already be done, but just for consistency.
    #We select optional = True for the semantic classes just in case, although it would be extremely unlikely for background to not be available...
    #We also include a semantic type parameter, which denotes what subtype of semantic class this belongs to (semantic seg is all "stuff").

    dataset_json["semantic_classes"] = semantic_labels_dict
    dataset_json["annotators"] = {
        "annotator_1":{
            "annotator_id":str(1),
            "annotation_protocol": dataset_json['reference']
        }
        }

    dataset_json["file_ext"] = ".nii.gz"
    dataset_json["channel_names"] = dict((v,k) for k,v in dataset_json["modality"].items())
    
    
    del dataset_json["labels"]
    del dataset_json["modality"]
    del dataset_json["training"]
    del dataset_json["test"]



    #We then reformat the training sets, test sets into the same structure so that it can be used downstream for any scripts or dataloaders which may want to perform
    #some kind of sampling (e.g., annotator-specific experiments, etc.)    
    
    #First we do the "training samples". We do not assume a fixed quantity of channels per case, but we do assume that the upper bound is fixed.

    training_samples_dict = dict() 

    for case_im_folder_path in extract_subdirs(target_imagesTr):
        case_name = full_path_splitter(case_im_folder_path)[-1]
        #Extracting case folder's relpaths for the images
        case_im_relpath = f'./{os.path.relpath(case_im_folder_path, target_folder)}' #This will give only the relative path from the "datasets" folder.
        #Extracting case folder's relpath for the annotations
        case_lb_relpath = f'./{os.path.relpath(os.path.join(target_labelsTr, case_name), target_folder)}'

        #We now construct the case dictionary, i.e., the relative paths for each of the relevant nifti files.
        case_dict = dict()

        ##################################################
        #Just handling the image paths 
        images_subdict = dict()
        #We then extract the relative paths for each of the image channels (we can assume that they are all present in this case since MSD does always provide them!)
        for channel, channel_code in dataset_json["channel_names"].items():
            images_subdict[channel] = os.path.join(case_im_relpath, case_name + "_%04.0d.nii.gz" % int(channel_code))
        ######################################
        
        #Just handling the annotation paths
        annotations_subdict = dict() 
        #We will now split the labels into the annotator -> semantic_class -> instances relpaths. We can assume that for MSD each annotator will always have a seg.
        #In fact we only have "annotator 1" as we were only provided with 1 annotation per sample.

        for annotator in dataset_json["annotators"].keys():
            annotator_subdict = dict()
            for semantic_lb in dataset_json["semantic_classes"].keys():
                semantic_class_subdict = dict() #Bit unnecessary but good to be somewhat consistent, this is a subdict which contains all the instances for a given 
                #semantic class. All the MSD dataset are semantic so only contain stuff, and we will make no strong assumptions a priori otherwise. In which case
                #instance num = 1 (it might even be empty/zeros!)
                for instance_idx, instance_path in enumerate(extract_subfiles(os.path.join(target_labelsTr, case_name, annotator, f"semantic_class_{semantic_lb}"))):
                    #checking the suffix to make sure the instance idx matches the filename just in case.
                    filename = file_ext_splitter(full_path_splitter(instance_path)[-1], dataset_json["file_ext"])[0]
                    #We check the number after the final _ evaluates to the same val as the instance id. 
                    instance_id = filename.split("_")[-1]
                    if int(instance_id) != instance_idx + 1:
                        raise Exception("Inconsistent notation for the instance ids")
                    semantic_class_subdict[f"instance_{instance_idx + 1}"] = os.path.join(case_lb_relpath, annotator, f"semantic_class_{semantic_lb}", filename + str(dataset_json["file_ext"]))   
                annotator_subdict[semantic_lb] = semantic_class_subdict 

            annotations_subdict[annotator] = annotator_subdict

        case_dict = {
            "images": images_subdict,
            "labels": annotations_subdict
        } 
        
        training_samples_dict[case_name] = case_dict 
    dataset_json["training"] = training_samples_dict 
    
    if process_Ts:
        #now for the test labels.
        test_samples_dict = dict()

        for case_im_folder_path in extract_subdirs(target_imagesTs):
            case_name = full_path_splitter(case_im_folder_path)[-1]
            #Extracting case folder's relpaths for the images
            case_im_relpath = f'./{os.path.relpath(case_im_folder_path, target_folder)}' #This will give only the relative path from the "datasets" folder.

            #We now construct the case dictionary, i.e., the relative paths for each of the relevant nifti files.
            case_dict = dict()

            ##################################################
            #Just handling the image paths 
            images_subdict = dict()
            #We then extract the relative paths for each of the image channels (we can assume that they are all present in this case since MSD does always provide them!)
            for channel, channel_code in dataset_json["channel_names"].items():
                images_subdict[channel] = os.path.join(case_im_relpath, case_name + "_%04.0d.nii.gz" % int(channel_code))
            
            case_dict["images"] = images_subdict 

            ######################################
        
            #Extracting case folder's relpath for the annotations
            case_lb_relpath = f'./{os.path.relpath(os.path.join(target_labelsTs, case_name), target_folder)}'
            #Just handling the annotation paths
            annotations_subdict = dict() 
            #We will now split the labels into the annotator -> semantic_class -> instances relpaths. We can assume that for MSD each annotator will always have a seg.
            #In fact we only have "annotator 1" as we were only provided with 1 annotation per sample.
        
            for annotator in dataset_json["annotators"].keys():
                annotator_subdict = dict()
                for semantic_lb in dataset_json["semantic_classes"].keys():
                    semantic_class_subdict = dict() #Bit unnecessary but good to be somewhat consistent, this is a subdict which contains all the instances for a 
                    #given semantic class. Most of the standard nnu-net like datasets are semantic so only contain stuff, and we will make no strong assumptions a 
                    # priori otherwise. In which case instance num = 1 (it might even be empty/zeros!)
                    for instance_idx, instance_path in enumerate(extract_subfiles(os.path.join(target_labelsTs, case_name, annotator, f"semantic_class_{semantic_lb}"))):
                        #checking the suffix to make sure the instance idx matches the filename just in case.
                        filename = file_ext_splitter(full_path_splitter(instance_path)[-1], dataset_json["file_ext"])[0]
                        #We check the number after the final _ evaluates to the same val as the instance id. 
                        instance_id = filename.split("_")[-1]
                        if int(instance_id) != instance_idx + 1:
                            raise Exception("Inconsistent notation for the instance ids")
                        semantic_class_subdict[f"instance_{instance_idx + 1}"] = os.path.join(case_lb_relpath, annotator, f"semantic_class_{semantic_lb}", filename + str(dataset_json["file_ext"]))   
                    annotator_subdict[semantic_lb] = semantic_class_subdict 

                annotations_subdict[annotator] = annotator_subdict

            case_dict["labels"] = annotations_subdict
        
            
        
        test_samples_dict[case_name] = case_dict 
    dataset_json["test"] = test_samples_dict 

    save_json(dataset_json, os.path.join(datasets_path, target_dataset_name, 'dataset.json'), sort_keys=False)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d_path', type=str, required=True,
    #                     help='Downloaded and extracted dataset folder absolute path. Example: '
    #                          '/home/parhomesmaeili/Radiology_Datasets/ToothFairy3')
    # parser.add_argument('-overwrite_id', type=int, required=True,
    #                     help='Select the dataset id. Unlike MSD which has a standardised id, most generic datasets will not, we recommended to determine your own convention!')
    # parser.add_argument('-input_file_ext', type=str, required=False, default='.nii.gz',
    #                     help='The file extension for the input files, currently assumed to be consistent across all image/annotation data.')
    # parser.add_argument('-output_file_ext', type=str, required=False, default='.nii.gz')
    # parser.add_argument('-process_Ts', action='store_true', default=False)
    # parser.add_argument('-np', type=int, required=False, default=4,
    #                     help=f'Number of processes used. Default: 1')
    # args = parser.parse_args()

    dataset_path = '/home/parhomesmaeili/Radiology_Datasets/ToothFairy3'
    convert_nnunet_style_dataset(source_folder=dataset_path, 
                                dataset_target_id=1000,
                                input_file_ext='.nii.gz',
                                output_file_ext='.nii.gz',
                                process_Ts=False,
                                num_processes=8)
    # convert_nnunet_style_dataset(args.i, args.dataset_target_id, args.input_file_ext, args.output_file_ext, args.process_Ts, args.np)
