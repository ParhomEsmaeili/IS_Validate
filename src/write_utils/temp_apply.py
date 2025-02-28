import torch
import nibabel as nib
from monai.transforms import Orientation, SaveImage, Orientationd, LoadImaged, EnsureTyped, EnsureChannelFirstd, Compose 
import copy 
from monai.data import MetaTensor 
import itk 
import numpy as np 
from typing import Union 

def write_itk(image_np, output_file, affine, dtype, compress):
    if isinstance(image_np, torch.Tensor):
        image_np = image_np.numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.numpy()
    if len(image_np.shape) >= 2:
        image_np = image_np.transpose().copy()
    if dtype:
        image_np = image_np.astype(dtype)

    result_image = itk.image_from_array(image_np)
    
    if affine is not None:
        
        convert_aff_mat = np.diag([-1, -1, 1, 1])
        if affine.shape[0] == 3:  # Handle RGB (2D Image)
            convert_aff_mat = np.diag([-1, -1, 1])
        affine = convert_aff_mat @ affine

        dim = affine.shape[0] - 1
        _origin_key = (slice(-1), -1)
        _m_key = (slice(-1), slice(-1))

        origin = affine[_origin_key]
        spacing = np.linalg.norm(affine[_m_key] @ np.eye(dim), axis=0)
        direction = affine[_m_key] @ np.diag(1 / spacing)


        result_image.SetDirection(itk.matrix_from_array(direction))
        result_image.SetSpacing(spacing)
        result_image.SetOrigin(origin)

    itk.imwrite(result_image, output_file, compress)

def converter(img:Union[torch.Tensor, MetaTensor], reference:MetaTensor):
    duplicate_reference = copy.deepcopy(reference)
    duplicate_img = copy.deepcopy(img)

    if isinstance(duplicate_img, MetaTensor):
        #Extract array
        array = duplicate_img.array 
        duplicate_reference.array = array 

    elif isinstance(duplicate_img, torch.Tensor):
        #If torch tensor.
        duplicate_reference.array = duplicate_img 
    
    return duplicate_reference 

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

    pred_fe_metatensor = converter(pred_fe_metatensor, loaded_copy)
    pred_fe_tensor = converter(pred_fe_tensor, loaded_copy)
    
    original_axcodes = nib.aff2axcodes(loaded_copy.meta['original_affine']) 
    
    orientation_undone_fn = Orientation(axcodes=original_axcodes)

    with orientation_undone_fn.trace_transform(False):
        orientation_undone_metatensor = orientation_undone_fn(pred_fe_metatensor)
        orientation_undone_constructed_metatensor = orientation_undone_fn(pred_fe_tensor) 


    #Now lets try with float datatypes:

    loaded_img_copy = copy.deepcopy(loaded_dict['image'])
    #For a dummy example where a MetaTensor object is provided.
    logits_fe_metatensor = loaded_img_copy * 10 #.array) 
    logits_fe_tensor = torch.from_numpy(loaded_img_copy.array)  * 100 #We just adjust the value for assurance in debugging.
    
    #Passing it through the converter.

    logits_fe_metatensor = converter(logits_fe_metatensor, loaded_img_copy)
    logits_fe_tensor = converter(logits_fe_tensor, loaded_img_copy)


    original_img_axcodes = nib.aff2axcodes(loaded_img_copy.meta['original_affine']) 
    
    orientation_undone_img_fn = Orientation(axcodes=original_img_axcodes)

    with orientation_undone_img_fn.trace_transform(False):
        orientation_undone_logits = orientation_undone_img_fn(logits_fe_metatensor)
        orientation_undone_constructed_logits = orientation_undone_img_fn(logits_fe_tensor) 


    # save_im = SaveImage(writer="ITKWriter")
    # save_im(orientation_undone_metatensor, {'affine':loaded_dict['image_meta_dict']['original_affine']}, '/home/parhomesmaeili/sanity_check_images/vscodeMetaTensorHandler.nii.gz')

    if not True:
        save_im(orientation_undone_constructed_metatensor, {'affine':loaded_dict['image_meta_dict']['original_affine']}, '/home/parhomesmaeili/sanity_check_images/vscodeConstructedMetaTensorHandler.nii.gz')
    else:
        write_itk(orientation_undone_metatensor.array[0], 
                output_file='/home/parhomesmaeili/sanity_check_images/pancreas/vscodeMetaTensorHandlerITKManual.nii.gz', 
                affine=loaded_copy.meta['original_affine'], 
                dtype=np.int32, 
                compress=False)
        
        write_itk(orientation_undone_constructed_metatensor.array[0], 
                output_file='/home/parhomesmaeili/sanity_check_images/pancreas/vscodeConstructedMetaTensorHandlerITKManual.nii.gz', 
                affine=loaded_copy.meta['original_affine'], 
                dtype=np.int32, 
                compress=False)
    print('halt!')

    #Writing dummy logits
    if True:
        write_itk(orientation_undone_logits.array[0], 
                output_file='/home/parhomesmaeili/sanity_check_images/pancreas/vscodeLogitsMetaTensorHandlerITKManual.nii.gz', 
                affine=loaded_img_copy.meta['original_affine'], 
                dtype=np.float32, 
                compress=False)
        
        write_itk(orientation_undone_constructed_logits.array[0], 
                output_file='/home/parhomesmaeili/sanity_check_images/pancreas/vscodeLogitsConstructedMetaTensorHandlerITKManual.nii.gz', 
                affine=loaded_img_copy.meta['original_affine'], 
                dtype=np.float32, 
                compress=False)
        
        #Writing an actual continuous map.

        write_itk(np.random.randn(*orientation_undone_constructed_logits.array.shape[1:]),
            output_file='/home/parhomesmaeili/sanity_check_images/pancreas/vscodeContinuousLogitsITKManual.nii.gz',
            affine=loaded_img_copy.meta['original_affine'],
            dtype=np.float32,
            compress=False
        )
