import torch
import nibabel as nib
from monai.transforms import Orientation, SaveImage, Orientationd, LoadImaged, EnsureTyped, EnsureChannelFirstd, Compose 
import copy 
from monai.data import MetaTensor 
import itk 
import tempfile 
import numpy as np 
from typing import Union 
from post import WriteOutput
import shutil

if __name__ == '__main__':
    # data_dict = {'image':'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTr/BraTS2021_01242.nii.gz',
    #             'label':'/home/parhomesmaeili/IS-Validation-Framework/IS_Validate/datasets/BraTS2021_Training_Data_Split_True_proportion_0.8_channels_t2_resized_FLIRT_binarised/imagesTr/labels/final/BraTS2021_01242.nii.gz'}

    data_dict = {
        'image':'/home/parhomesmaeili/Radiology_Datasets/PRISM Datasets/Task07_Pancreas/Task07_Pancreas/Training/pancreas_001/image.nii.gz',
        'label':'/home/parhomesmaeili/Radiology_Datasets/PRISM Datasets/Task07_Pancreas/Task07_Pancreas/Training/pancreas_001/segmentation.nii.gz'
        }
    load_transforms = [
        LoadImaged(keys=['image', 'label'], reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        EnsureTyped(keys=['image', 'label'], dtype=[torch.float64, torch.int64]),
    ]

    compose_transf = Compose(load_transforms) 

    loaded_dict = compose_transf(data_dict)

    #We move the affines all onto tensors for the sake of consistency...
    loaded_dict['image'].meta['original_affine'] = torch.from_numpy(loaded_dict['image'].meta['original_affine'])
    loaded_dict['label'].meta['original_affine'] = torch.from_numpy(loaded_dict['label'].meta['original_affine'])

    #First we copy the MetaTensor object so we can store the information about the affine matrix it is currently assuming. 
    loaded_copy = copy.deepcopy(loaded_dict['label'])

    #For a dummy example where a MetaTensor object is provided.
    pred_fe_metatensor = copy.deepcopy(loaded_dict['label']) * 10 #.array) 
    pred_fe_tensor = torch.from_numpy(copy.deepcopy(loaded_dict['label'].array))  * 10 #We just adjust the label value for assurance in debugging.

    #For an example where a torch tensor is provided instead.
    # if not True:
    
    # constructed_metatensor = copy.deepcopy(loaded_dict['label']) 
    # constructed_metatensor.array = pred_fe_array   
    
    # else:
    #     constructed_metatensor = copy.deepcopy(loaded_dict['label'].array) * 10 #We extract the array and just adjust the label value for assurance in debugging.

    # pred_fe_metatensor = converter(pred_fe_metatensor, loaded_copy)
    # pred_fe_tensor = converter(pred_fe_tensor, loaded_copy)
    
    # original_axcodes = nib.aff2axcodes(loaded_copy.meta['original_affine']) 
    
    # orientation_undone_fn = Orientation(axcodes=original_axcodes)

    # with orientation_undone_fn.trace_transform(False):
    #     orientation_undone_metatensor = orientation_undone_fn(pred_fe_metatensor)
    #     orientation_undone_constructed_metatensor = orientation_undone_fn(pred_fe_tensor) 


    #Now lets try with float datatypes:

    loaded_img_copy = copy.deepcopy(loaded_dict['image'])
    #For a dummy example where a MetaTensor object is provided.
    probs_fe_metatensor = torch.cat([loaded_img_copy * n for n in range(5)])   #.array) 
    probs_fe_tensor = torch.cat([torch.from_numpy(loaded_img_copy.array)  * 10 * n for n in range(5)]) #We just adjust the value for assurance in debugging.
    
    #Passing it through the converter.

    # probs_fe_metatensor = converter(probs_fe_metatensor, loaded_img_copy)
    # probs_fe_tensor = converter(probs_fe_tensor, loaded_img_copy)


    # original_img_axcodes = nib.aff2axcodes(loaded_img_copy.meta['original_affine']) 
    
    # orientation_undone_img_fn = Orientation(axcodes=original_img_axcodes)

    # with orientation_undone_img_fn.trace_transform(False):
    #     orientation_undone_probs = orientation_undone_img_fn(probs_fe_metatensor)
    #     orientation_undone_constructed_probs = orientation_undone_img_fn(probs_fe_tensor) 


    # save_im = SaveImage(writer="ITKWriter")
    # save_im(orientation_undone_metatensor, {'affine':loaded_dict['image_meta_dict']['original_affine']}, '/home/parhomesmaeili/sanity_check_images/vscodeMetaTensorHandler.nii.gz')

    # if not True:
    #     save_im(orientation_undone_constructed_metatensor, {'affine':loaded_dict['image_meta_dict']['original_affine']}, '/home/parhomesmaeili/sanity_check_images/vscodeConstructedMetaTensorHandler.nii.gz')
    # else:
    #     write_itk(orientation_undone_metatensor.array[0], 
    #             output_file='/home/parhomesmaeili/sanity_check_images/pancreas/vscodeMetaTensorHandlerITKManual.nii.gz', 
    #             affine=loaded_copy.meta['original_affine'], 
    #             dtype=np.int32, 
    #             compress=False)
        
    #     write_itk(orientation_undone_constructed_metatensor.array[0], 
    #             output_file='/home/parhomesmaeili/sanity_check_images/pancreas/vscodeConstructedMetaTensorHandlerITKManual.nii.gz', 
    #             affine=loaded_copy.meta['original_affine'], 
    #             dtype=np.int32, 
    #             compress=False)
    # print('halt!')

    # #Writing dummy probs
    # if True:
    #     for i in range(5):
    #         write_itk(orientation_undone_probs.array[i], 
    #             output_file=f'/home/parhomesmaeili/sanity_check_images/pancreas/vscodeprobsMetaTensorHandlerITKManual_{i}.nii.gz', 
    #             affine=loaded_img_copy.meta['original_affine'], 
    #             dtype=np.float32, 
    #             compress=False)
            
    #         write_itk(orientation_undone_constructed_probs.array[i], 
    #             output_file=f'/home/parhomesmaeili/sanity_check_images/pancreas/vscodeprobsConstructedMetaTensorHandlerITKManual_{i}.nii.gz', 
    #             affine=loaded_img_copy.meta['original_affine'], 
    #             dtype=np.float32, 
    #             compress=False)
        
    #     #Writing an actual continuous map.

    #     write_itk(np.random.randn(*orientation_undone_constructed_probs.array.shape[1:]),
    #         output_file='/home/parhomesmaeili/sanity_check_images/pancreas/vscodeRandomContinuousprobsITKManual.nii.gz',
    #         affine=loaded_img_copy.meta['original_affine'],
    #         dtype=np.float32,
    #         compress=False
    #     )

    #Create the tempdir for testing:
    dummy_tempdir = tempfile.TemporaryDirectory(dir='/home/parhomesmaeili/sanity_check_images/tmp')

    pred_output_writer = WriteOutput('label', 'pred', dtype=np.int32, compress=False, invert_orient=True)
    probs_output_writer = WriteOutput('image', 'probs', dtype=np.float32, compress=False, invert_orient=True)

    data_instance = {'image': {
            'path': data_dict['image'],
            'metatensor': loaded_dict['image'],
            'meta_dict': loaded_dict['image'].meta
            },
        'label': {
            'path': data_dict['label'],
            'metatensor': loaded_dict['label'],
            'meta_dict': loaded_dict['label'].meta 
        }
        }

    output_metatensor_format = {
        'pred': {
            'metatensor': pred_fe_metatensor,
            'meta_dict': {}
            },
        'probs':{
            'metatensor': probs_fe_metatensor,
            'meta_dict':{}
        }
    }

    output_tensor_format = {
        'pred': {
            'metatensor': pred_fe_tensor,
            'meta_dict': {}
            },
        'probs':{
            'metatensor': probs_fe_tensor,
            'meta_dict':{}
        }
    }
    
    pred_metatensor_paths = pred_output_writer(
        data_instance=data_instance,
        output_data=output_metatensor_format,
        tmp_dir=dummy_tempdir.name
        )
    probs_metatensor_paths = probs_output_writer(
        data_instance=data_instance,
        output_data=output_metatensor_format,
        tmp_dir=dummy_tempdir.name
    )

    shutil.move(pred_metatensor_paths[0], '/home/parhomesmaeili/sanity_check_images/pancreas/vscodeMetaTensorHandlerITKManual.nii.gz')
    for i in range(len(probs_metatensor_paths)):
        shutil.move(probs_metatensor_paths[i], f'/home/parhomesmaeili/sanity_check_images/pancreas/vscodeprobsMetaTensorHandlerITKManual_{i}.nii.gz')

    print('halt, now the plain tensors.')


    pred_metatensor_paths = pred_output_writer(
        data_instance=data_instance,
        output_data=output_tensor_format,
        tmp_dir=dummy_tempdir.name
        )
    probs_metatensor_paths = probs_output_writer(
        data_instance=data_instance,
        output_data=output_tensor_format,
        tmp_dir=dummy_tempdir.name
    )

    shutil.move(pred_metatensor_paths[0], '/home/parhomesmaeili/sanity_check_images/pancreas/vscodeConstructedMetaTensorHandlerITKManual.nii.gz')
    for i in range(len(probs_metatensor_paths)):
        shutil.move(probs_metatensor_paths[i], f'/home/parhomesmaeili/sanity_check_images/pancreas/vscodeprobsConstructedMetaTensorHandlerITKManual_{i}.nii.gz')
