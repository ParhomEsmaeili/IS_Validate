import logging
from typing import Optional, Sequence, Union
import nibabel as nib
import numpy as np
# import skimage.measure as measure
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Orientation
)
import itk 
import tempfile 
import copy 


class WriteOutput:
    def __init__(
        self,
        ref_key: str,
        result_key: str,
        dtype:type = None,
        compress:bool=False,
        invert_orient: bool = True,
        file_ext:str = '.nii.gz'
    ):
        # super().__init__(keys)
        self.ref_key = ref_key
        self.result_key = result_key
        self.dtype = dtype 
        self.compress = compress 
        self.invert_orient = invert_orient
        self.file_ext = file_ext 

    # def is_channelfirst_image(self, image: MetaTensor) -> bool:
    #     """Check if the provided image contains channel dimensions

    #     Args:
    #         image: Expected shape (channels, width, height, batch)

    #     Returns:
    #         bool: If this is a channel first tensor or not
    #     """
    #     #We extract the number of spatial dimensions from the metainformation.
    #     n_spatial_dims = int(image.meta['dim[0]'])
        
    #     if not len(image.shape) == n_spatial_dims + 1:
    #         raise Exception('The expected dimensions were that of a channel first image.')
        

    def write_itk(self, image_np, affine, tmp_dir): #, dtype, compress): #output_file, affine, dtype, compress):
        
        output_file = tempfile.NamedTemporaryFile(suffix=self.file_ext, delete=False, dir=tmp_dir).name

        if isinstance(image_np, torch.Tensor):
            image_np = image_np.numpy()
        if isinstance(affine, torch.Tensor):
            affine = affine.numpy()
        if len(image_np.shape) >= 2:
            image_np = image_np.transpose().copy()
        if self.dtype:
            image_np = image_np.astype(self.dtype)

        result_image = itk.image_from_array(image_np)
        
        if affine is not None:
            
            convert_aff_mat = np.diag([-1, -1, 1, 1])
            if affine.shape[0] == 3:
                raise NotImplementedError('We do not yet provide handling for 2D images')
                # if affine.shape[0] == 3:  # Handle RGB (2D Image)
                    # convert_aff_mat = np.diag([-1, -1, 1])

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

        itk.imwrite(result_image, output_file, self.compress)

        return output_file 
    
    def converter(self, img:Union[torch.Tensor, MetaTensor], reference:MetaTensor):
        duplicate_reference = copy.deepcopy(reference)
        duplicate_img = copy.deepcopy(img)

        if isinstance(duplicate_img, MetaTensor):
            #Extract array
            array = duplicate_img.array 
            duplicate_reference.array = array 

        elif isinstance(duplicate_img, torch.Tensor):
            #If torch tensor.
            duplicate_reference.array = duplicate_img 
        else:
            raise TypeError(f'The output {self.result_key} must be a MetaTensor or a torch Tensor.')
        
        return duplicate_reference 
    
    def __call__(self, data_instance: dict, output_data: dict, tmp_dir: str):
        
        ref = data_instance[self.ref_key]['metatensor']
        ref_meta_dict = ref.meta
        
        result = output_data[self.result_key]['metatensor']

        result_reformat = self.converter(result, ref)

        if self.invert_orient:
            # Undo Orientation
            orig_affine = ref_meta_dict.get("original_affine", None)
            if orig_affine is not None:
                orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
                inverse_transform = Orientation(axcodes=orig_axcodes)
                # Apply inverse
                with inverse_transform.trace_transform(False):
                    result_reformat = inverse_transform(result_reformat)
            else:
                raise Exception("Failed invert orientation - original_affine is not on the image header")
            
        #We assume that we have already checked that the outputs are channel first, and that they meet the quantity of channels.

        #Add a line of code for extracting the channels, this also removes the channel dimension.
        channel_split = list(result_reformat.array) 

        #Make use of the writeitk and call it across each channel
        path_list = [self.write_itk(channel, ref_meta_dict["original_affine"], tmp_dir) for channel in channel_split]

        #Return the list of paths for all of the temp files.
        return path_list
