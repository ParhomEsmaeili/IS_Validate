#Intended for the abstract functions utilised in generating bbox prompts.
import os 
from os.path import dirname as up
import sys
import torch 
sys.path.append(up(up(up(up(up(os.path.abspath(__file__)))))))

def planar_bbox_from_binary_mask(binary_mask: torch.Tensor, args:dict):
    '''
    Function which generates a 2D bounding box from an input binary mask.
    '''
    pass

def volumetric_bbox_from_binary_mask(binary_mask: torch.Tensor, args:dict):
    '''
    Function which generates a 3D bounding box from an input binary mask.
    '''
    pass 

def jitter(bbox: torch.Tensor, volume_shape: torch.Size, jitter_threshold: list[int]):
    '''
    Function which applies a uniform random sampling from INTEGERS to jitter the bounding box coordinates. I.e., it
    will randomly perturb them along voxels.
    
    inputs: 

    bbox: A 2D or 3D bounding box, represented as a tensor of shape (1, 2 * D) where D is the number of spatial dimensions
    in an image. D = 3 ALWAYS. ASSUMPTION: bbox is in a valid region (i.e. it is not degenerate and within the image bounds).
    jitter_threshold: A list of int representing the upper limit for uniform sampling jitter to apply to the bounding box.


    requirement: Jitter_threshold should match the number of dimensions. 
    '''
    #Lets check the dimensionality of the bbox, we will do this by checking whether the min-max pairs are the same.
    if len(jitter_threshold) != 3:
        raise NotImplementedError("Support is only provided for bounding boxes in volumetric images")
    else:
        matching_dims = [i for i in range(bbox.shape[1] // 2) if bbox[0, i] == bbox[0, i + 3]]
        if len(matching_dims) > 1:
            raise ValueError("Cannot have a bounding box where more than one dimension has zero thickness.")
        else:
            if len(jitter_threshold) != bbox.shape[1] // 2:
                raise ValueError("Jitter threshold length must match the number of dimensions in the bounding box.")
            #In this case we have a potentially valid bbox. We will now apply jitter. 
            
            #We will now sample a jitter for each dimension (if the threshold is 0 then obviously no jitter will be applied).
            jitter_values = torch.tensor([torch.randint(-thresh, thresh, (1,)).item() if thresh != 0 else 0 for thresh in jitter_threshold] * 2)
            jitter_values = jitter_values.unsqueeze(0)  #Make it a 2D tensor with shape (1, 2*D)

            #Now we will apply the jitter to the bbox.
            bbox_jittered = bbox.clone()
            bbox_jittered += jitter_values 

            #Now we just need to ensure the bbox is still within the volume bounds. Truncate if necessary.
            for i in range(bbox.shape[1] // 2):
                min_coord = max(bbox_jittered[0, i], 0)
                max_coord = min(bbox_jittered[0, i + 3], volume_shape[i] - 1)
                if min_coord > max_coord:
                    raise Exception("Not possible for min coord to be greater than max coord after jittering. Check implementation.")
            
            #Now we will check whether the bbox is still valid. It may be possible that jitter + clamping has made it
            #invalid. If so, we will just reset to the original bbox. ASSUMPTION: The original bbox is valid.

            return bbox_jittered




def bbox_jittered(binary_mask: torch.Tensor):
    '''
    Function which computes a jittered bounding box from the input binary mask.
    '''
    bbox_extrema = bbox_extrema(binary_mask)

def bbox_extrema(binary_mask: torch.Tensor):
    '''
    Helper function which computes the bounding box extrema from a binary mask. 

    inputs: binary_mask: A binary mask tensor, can have shape (D, H, W) or (H, W) depending on whether 3D or 2D.
    outputs: bbox: A tensor of shape (1, 2*D) representing the bounding box extrema in the format [min_x, min_y, min_z, max_x, max_y, max_z]
    for 3D and [min_x, min_y, max_x, max_y] for 2D.    
    '''
    if binary_mask.dtype != torch.bool:
        raise TypeError("Input binary mask must be of boolean type.")
    
    dims = binary_mask.dim()
    if dims == 3:
        #3D case
        positions = torch.nonzero(binary_mask, as_tuple=False)
        min_x, min_y, min_z = torch.min(positions, dim=0)[0]
        max_x, max_y, max_z = torch.max(positions, dim=0)[0]
        bbox = torch.tensor([[min_x.item(), min_y.item(), min_z.item(), max_x.item(), max_y.item(), max_z.item()]])
        return bbox
    elif dims == 2:
        #2D case
        positions = torch.nonzero(binary_mask, as_tuple=False)
        min_x, min_y = torch.min(positions, dim=0)[0]
        max_x, max_y = torch.max(positions, dim=0)[0]
        bbox = torch.tensor([[min_x.item(), min_y.item(), max_x.item(), max_y.item()]])
        return bbox

def component_selection(binary_mask: torch.Tensor, component_selection_type: str):
    '''
    Function which selects connected components from the input binary mask based on the provided selection type.
    Given that bbox is a partition-based prompting mechanism.
    '''
    pass

def planar_extraction(binary_mask: torch.Tensor):
    '''
    Function which extracts a 2d bbox.
    '''
    pass

def volumetric_extraction(binary_mask: torch.Tensor):
    '''
    Function which extracts a 3d bbox.
    '''
    pass 
