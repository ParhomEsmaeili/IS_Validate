#Borrowing skeleton from nnU-Net convert_MSD_dataset.py, with some modifications for our own requirements. Only intended for re-structuring semantic segmentation
#datasets. Panoptic, and instance segmentation as a subcategory will have to adhere to a structure provided (and it is not commonly used enough yet to have a good format!)

import argparse
import multiprocessing
import shutil
from typing import Optional
import SimpleITK as sitk
import os 
import sys 
import numpy as np
import json
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
    standard_dataset_json_gen)

def split_semantic_seg_nifti(filename, output_folder, semantic_class_dict):
    '''
    Function intended for splitting 3D semantic segmentation volumes into separate binary class segmentations, and writing them in nifti format:
    
    filename = case_name
    output_folder = path to segmentation directory for the given case
    semantic_class_dict = dictionary containing semantic class strings: integer codes (str or int type)

    Pretty standard as the vast majority of research datasets are presented as semantic segmentation datasets.
    '''
    #We partition semantic class by folders, therefore we require that the output folder is automatically generated.
    os.makedirs(output_folder, exist_ok=False)

    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = os.path.basename(filename)
    
    #We presume only 3D volumes for the segmentations! Purely spatial-component assumption (4D volumes for images are assumed to be co-registered)

    if dim != 3:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, filename))
    else:
        seg_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = img_itk.GetDirection()
        #We partition according to semantic class codes in the semantic_class_dict. Within each we remap to a "single instance", we do not assume distinct id's 
        #for "stuff" classes but use the same convention for consistency.
        for code, word in semantic_class_dict.items():
            if isinstance(code, int):
                seg_class_word = np.where(seg_npy == code, 1, 0)
            elif isinstance(code, str): 
                code = int(code) #Throws an error if the code is a float, it must be an int even if expressed as a str.
                seg_class_word = np.where(seg_npy == code, 1, 0)
            
            seg_itk_new = sitk.GetImageFromArray(seg_class_word)
            seg_itk_new.SetSpacing(spacing)
            seg_itk_new.SetOrigin(origin)
            seg_itk_new.SetDirection(direction)
            #Creating the subfolder for each semantic class for each sample.
            semantic_class_subfolder = os.path.join(output_folder, f'semantic_class_{word}')
            os.makedirs(semantic_class_subfolder, exist_ok=False)
            sitk.WriteImage(seg_itk_new, os.path.join(semantic_class_subfolder, file_base[:-7] + "_%04.0d.nii.gz" % 1)) 
            #We hardcode the instance id = 1 because for semantic segmentation datasets each class a-priori is assumed only to have a single "instance" mask.
  

def split_4d_MSD_nifti(filename, output_folder):
    #We partition by folders, therefore we require that the output folder is automatically generated.
    #Assumption: The quantity of input channels is fixed across all samples in a dataset (valid for MSD)
    os.makedirs(output_folder, exist_ok=False)

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
        for i, t in enumerate(range(img_npy.shape[0])):
            img = img_npy[t]
            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(spacing)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(img_itk_new, os.path.join(output_folder, file_base[:-7] + "_%04.0d.nii.gz" % i))

def convert_msd_dataset(source_folder: str, overwrite_target_id: Optional[int] = None,
                        process_labelsTs: bool = False, 
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
    if process_labelsTs:
        assert os.path.isdir(labelsTs_path) and os.path.exists(labelsTs_path), "labelsTs missing in source folder or was not a subdirectory."
    
    dataset_json = os.path.join(source_folder, 'dataset.json')
    assert os.path.isfile(dataset_json), f"MSD formatted dataset.json was missing in source_folder"

    # Extract the dataset id and name: MSD uses convention: TaskXX_YYYY where YYYY=Task name (broadly)
    task, dataset_name = os.path.basename(source_folder).split('_')
    task_id = int(task[4:]) #We assume a fixed config for the dataset naming convention for MSD here.

    # check if target dataset id is already being used for another dataset.
    target_id = task_id if overwrite_target_id is None else overwrite_target_id
    existing_datasets = check_dataset_existence(datasets_path, target_id)
    assert len(existing_datasets) == 0, f"Target dataset id {target_id} is already taken, please consider changing " \
                                        f"it using overwrite_target_id. Conflicting dataset: {existing_datasets}"

    target_dataset_name = f"Dataset{target_id:03d}_{dataset_name}"
    target_folder = os.path.join(datasets_path, target_dataset_name)
    target_imagesTr = os.path.join(target_folder, 'imagesTr')
    target_imagesTs = os.path.join(target_folder, 'imagesTs')
    target_labelsTr = os.path.join(target_folder, 'labelsTr')
    target_labelsTs = os.path.join(target_folder, 'labelsTs')
    
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_imagesTs, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)
    if process_labelsTs:
        os.makedirs(target_labelsTs, exist_ok=True) 

    #Loading the original dataset json, will be reformatted but also used for splitting semantic seg. labels.
    dataset_json = load_json(dataset_json)

    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []

        # convert 4d train images
        source_images = [i for i in extract_nifti_files(imagesTr_path, join=False, sort=True) if
                         not i.startswith('.') and not i.startswith('_')]
        #appending the filename to the folder path.
        source_images = [os.path.join(imagesTr_path, i) for i in source_images]
        
        #We partition the dataset into the "samples" folders, i.e., whatever the definition the MSD-formatted dataset uses to stratify images into a 
        # given sample. This is with the assumption that these would come as 4D volumes. Other datasets, of course, do not assume this. What one can mean by a given
        # "sample" depends on the context of the dataset. 

        # If using one machine it might be more straightforward: it might be different series' within a single study. With different machines these might be images 
        # from a reasonably short time interval, one assumes that they are are co-registered a priori, etc. It all depends on the dataset! We just assume that the 
        # spatial dimensions & voxel counts will be identical across each 3D volume in the 4D volume. The channel will contain whatever is required.
        
        #We perform this partition to correspond with the structure used for the annotations being used, for consistency. And also for readability.

        target_sample_folders = [os.path.join(target_imagesTr, file_ext_splitter(full_path_splitter(i)[-1],suffix='.nii.gz')[0]) for i in source_images]
        # split_4d_MSD_nifti(source_images[0], target_sample_folders[0])
        results.append(
            p.starmap_async(
                split_4d_MSD_nifti, zip(source_images, target_sample_folders) #[target_imagesTr] * len(source_images))
            )
        )

        # convert 4d test images
        source_images = [i for i in extract_nifti_files(imagesTs_path, join=False, sort=True) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [os.path.join(imagesTs_path, i) for i in source_images]
        target_sample_folders = [os.path.join(target_imagesTs, file_ext_splitter(full_path_splitter(i)[-1],suffix='.nii.gz')[0]) for i in source_images]
        results.append(
            p.starmap_async(
                split_4d_MSD_nifti, zip(source_images, target_sample_folders)
            )
        )

        # convert training segmentations
        source_images = [i for i in extract_nifti_files(labelsTr_path, join=False, sort=True) if
                         not i.startswith('.') and not i.startswith('_')]
        source_images = [os.path.join(labelsTr_path, i) for i in source_images]
        #For MSD, in this script, we will just assume it is fully semantic segmentation. Not worth performing label remappings for segmentation tasks which are not 
        # fully disjoint in the loop of a dataloader, as it would be preferable to double check the outputs & because we don't have any good strategies for robustly
        # splitting semantic masks into instance or panoptic masks without connected component analysis (which is not good enough for all cases!) 
        #
        # That is an open problem?..
        
        #MSD is single-annotator (only one annotation provided or pre-fused), instance_id = 1 for all semantic classes.  
        
        # for s in source_images:
        #     shutil.copy(os.path.join(labelsTr_path, s), os.path.join(target_labelsTr, s))
        target_sample_folders = [os.path.join(target_labelsTr, file_ext_splitter(full_path_splitter(i)[-1],suffix='.nii.gz')[0], 'annotator_1') for i in source_images]
        results.append(
            p.starmap_async(
                split_semantic_seg_nifti, zip(source_images, target_sample_folders, [dataset_json['labels']] * len(source_images))
            )
        )

        #We add the option (in case the folder exists) to also process LabelsTs. We presume that this folder will have the same structure as LabelsTr in relation to 
        #imagesTs.
        if process_labelsTs:
            source_images = [i for i in extract_nifti_files(labelsTs_path, join=False, sort=True) if
                         not i.startswith('.') and not i.startswith('_')]
            source_images = [os.path.join(labelsTs_path, i) for i in source_images]

            target_sample_folders = [os.path.join(target_labelsTs, file_ext_splitter(full_path_splitter(i)[-1],suffix='.nii.gz')[0], 'annotator_1') for i in source_images]
            # split_semantic_seg_nifti(source_images[0], target_sample_folders[0], dataset_json['labels'])
            results.append(
                p.starmap_async(
                    split_semantic_seg_nifti, zip(source_images, target_sample_folders, [dataset_json['labels']] * len(source_images))
                )
            )
        
        [i.get() for i in results]

    
    #Remapping the dataset configs into an nnU-Net inspired style for the channel naming convention. 
    # Appending the file extension to the set of configs for downstream functions to be able to exploit. 
    # (We may later require some adaptation to nrrd files for instance segmentation/panoptic seg. labels)

    # dataset_json["semantic_classes"] = semantic_labels_dict

    channel_names = dict((v,k) for k,v in dataset_json["modality"].items())
    annotator_descrip = {
        "annotator_1":{
            "annotator_id":str(1),
            "annotation_protocol": "https://arxiv.org/pdf/1902.09063"
        }
    }
    # dataset_json["annotators"] = annotator_descript
    
    # dataset_json["file_ext"] = ".nii.gz"
    # dataset_json["channel_names"] = channel_names #dict((v,k) for k,v in dataset_json["modality"].items())
    
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
    #We also include a semantic type parameter, which denotes what subtype of semantic class this belongs to (semantic seg is all "stuff" by default)

    file_ext = '.nii.gz'

    #We then reformat the training sets, test sets into the same structure so that it can be used downstream for any scripts or dataloaders which may want to perform
    #some kind of sampling (e.g., annotator-specific experiments, etc.)    
    
    #First we do the "training samples". Since we assume the MSD dataset here, we can safely assume that the number of channels will be consistent across all samples.

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
        for channel, channel_code in channel_names.items():
            images_subdict[channel] = os.path.join(case_im_relpath, case_name + "_%04.0d" % int(channel_code) + str(file_ext))
        ######################################
        #Just handling the annotation paths
        annotations_subdict = dict() 
        #We will now split the labels into the annotator -> semantic_class -> instances relpaths. We can assume that for MSD each annotator will always have a seg.
        #In fact we only have "annotator 1" as we were only provided with 1 annotation per sample.

        for annotator in annotator_descrip.keys():
            annotator_subdict = dict()
            for semantic_lb in semantic_labels_dict.keys():

                semantic_class_subdict = dict() #Bit unnecessary but good to be somewhat consistent, this is a subdict which contains all the instances for a given 
                #semantic class. All the MSD dataset are semantic so only contain stuff, and we will make no strong assumptions a priori otherwise. In which case
                #instance num = 1 (it might even be empty/zeros!)
                for instance_idx, instance_path in enumerate(extract_subfiles(os.path.join(target_labelsTr, case_name, annotator, f"semantic_class_{semantic_lb}"))):
                    #checking the suffix to make sure the instance idx matches the filename just in case.
                    filename = file_ext_splitter(full_path_splitter(instance_path)[-1], file_ext)[0]
                    #We check the number after the final _ evaluates to the same val as the instance id. 
                    instance_id = filename.split("_")[-1]
                    if int(instance_id) != instance_idx + 1:
                        raise Exception("Inconsistent notation for the instance ids")
                    semantic_class_subdict[f"instance_{instance_idx + 1}"] = os.path.join(case_lb_relpath, annotator, f"semantic_class_{semantic_lb}", filename + str(file_ext))   
                annotator_subdict[semantic_lb] = semantic_class_subdict 

            annotations_subdict[annotator] = annotator_subdict

        case_dict = {
            "images": images_subdict,
            "labels": annotations_subdict
        } 
        
        training_samples_dict[case_name] = case_dict 
    
    
    # dataset_json["training"] = training_samples_dict 

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
        for channel, channel_code in channel_names.items():
            images_subdict[channel] = os.path.join(case_im_relpath, case_name + "_%04.0d" % int(channel_code) + file_ext)
        
        case_dict["images"] = images_subdict 

        ######################################
        
        if process_labelsTs:
            #Only if the labels are actually available. 

            #Extracting case folder's relpath for the annotations
            case_lb_relpath = f'./{os.path.relpath(os.path.join(target_labelsTs, case_name), target_folder)}'
            #Just handling the annotation paths
            annotations_subdict = dict() 
            #We will now split the labels into the annotator -> semantic_class -> instances relpaths. We can assume that for MSD each annotator will always have a seg.
            #In fact we only have "annotator 1" as we were only provided with 1 annotation per sample.
            for annotator in annotator_descrip.keys():
                annotator_subdict = dict()
                for semantic_lb in semantic_labels_dict.keys():
                    semantic_class_subdict = dict() 
                    #Bit unnecessary but good to be somewhat consistent, this is a subdict which contains all the instances for a given 
                    #semantic class. All the MSD dataset are semantic so only contain stuff, and we will make no strong assumptions a priori otherwise. In which case
                    #instance num = 1 (it might even be empty/zeros!)
                    for instance_idx, instance_path in enumerate(extract_subfiles(os.path.join(target_labelsTs, case_name, annotator, f"semantic_class_{semantic_lb}"))):
                        #checking the suffix to make sure the instance idx matches the filename just in case.
                        filename = file_ext_splitter(full_path_splitter(instance_path)[-1], file_ext)[0]
                        #We check the number after the final _ evaluates to the same val as the instance id. 
                        instance_id = filename.split("_")[-1]
                        if int(instance_id) != instance_idx + 1:
                            raise Exception("Inconsistent notation for the instance ids")
                        semantic_class_subdict[f"instance_{instance_idx + 1}"] = os.path.join(case_lb_relpath, annotator, f"semantic_class_{semantic_lb}", filename + str(file_ext))   
                    annotator_subdict[semantic_lb] = semantic_class_subdict 

                annotations_subdict[annotator] = annotator_subdict

            case_dict["labels"] = annotations_subdict
            
        test_samples_dict[case_name] = case_dict 

    
    # dataset_json["test"] = test_samples_dict 

    standard_dataset_json_gen(
        output_folder=os.path.join(datasets_path, target_dataset_name),
        annotator_dict=annotator_descrip,
        channel_names=channel_names,
        semantic_labels=semantic_labels_dict,
        num_training_cases=dataset_json['numTraining'],
        file_ext=file_ext,
        tensorImageSize=dataset_json['tensorImageSize'],
        reference=dataset_json['reference'],
        train_relpaths=training_samples_dict,
        num_test_cases=dataset_json['numTest'],
        test_relpaths=test_samples_dict,
        citation='http://medicaldecathlon.com/',
        dataset_name=dataset_json['name'],
        release=dataset_json['release'],
        description=dataset_json['description'],
        license=dataset_json['licence']        
    )
    
    # save_json(dataset_json, os.path.join(datasets_path, target_dataset_name, 'dataset.json'), sort_keys=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_path', type=str, required=True,
                        help='Downloaded and extracted MSD dataset folder absolute path. Example: '
                             '/home/parhomesmaeili/Radiology_Datasets/MSD/Task01_BrainTumour')
    parser.add_argument('-overwrite_id', type=int, required=False, default=None,
                        help='Overwrite the dataset id. If not set we use the id of the task (inferred from '
                             'folder name). Only use the defaults if you already have an static configuration for the dataset IDs, otherwise recommended to determine!'
                             'a common sense approach to structuring your datasets...')
    parser.add_argument('-process_labelsTs', action='store_true', default=False)
    parser.add_argument('-np', type=int, required=False, default=1,
                        help=f'Number of processes used. Default: 1')
    args = parser.parse_args()
    convert_msd_dataset(args.d_path, args.overwrite_id, args.process_labelsTs, args.np)