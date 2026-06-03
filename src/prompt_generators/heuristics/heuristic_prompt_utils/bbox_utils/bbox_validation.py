"""
Validation functions for checking bounding box validity.

This module is now focused exclusively on checking bbox extrema validity
(e.g., degenerate dimensions, out-of-bounds coordinates, dimensionality matching).
Spatial extent checks (contiguous sequence analysis) have been moved to
spatial_utils/spatial_extent.py.

"""
import torch
from typing import List, Tuple

def check_bbox_validity(
    bbox_extrema: torch.Tensor,
    context_config: dict,
    critical_failure: bool = False
):
    '''
    Function which checks the validity of a bounding box.
    
    inputs:
    bbox_extrema: A tensor of shape (1, 6) representing the extrema of the
    bounding box in the format [min_x, min_y, min_z, max_x, max_y, max_z]

    context_config: A dict which contains some information about the context in which the bbox is being generated.
        required fields:
            dimensionality: dict[
            expected_dimensionality: int, either 2 or 3, representing the expected dimensionality of the bounding box. This is required to understand how to check the validity of the bbox, e.g., whether to check for degenerate dimensions, and which dimensions to check for non-negativity and bounds.
            collapsed_dimension: int, if the expected dimensionality is 2D then this field specifies which dimension was collapsed.
            ]
            image_dimensions: torch.Size representing the dimensions of the image volume which the bbox is situated in
    critical_failure: bool, if True, then any invalidity in the bbox will be considered a critical failure and raise
    an exception. If false, then the function will return a bool indicating whether the bbox is valid, and the list of
    the dimensions which are invalid (if any).

    outputs:
    valid_bool: bool indicating whether the bbox is valid or not.
    invalid_indices: List of indices in the bbox extrema which are invalid or not, assuming a critical_failure is not
    raised.
    
    Some measures for bbox validity: 
        1) min_x <= max_x, min_y <= max_y, min_z <= max_z
        2) All coordinates must be non-negative (assuming the origin is at the corner of the image and the bbox is within the image)
        3) All coordinates must be within the bounds of the image dimensions (this would require passing the image dimensions to this function, or ensuring that the jitter parameters are generated in such a way that this is guaranteed).
        4) The bbox should not be degenerate, and it should not change in dimensionality
            -> this is cross-referenced with the expected_dimensionality key in the context config. 
            a) A 2D bbox should not become a 3D bbox, or a point/line
            b) A 3D bbox should not become a 2D bbox, or a point/line
    
    '''
    assert 'dimensionality' in context_config, "Context config must contain the key 'dimensionality' to specify whether the details on the bounding box"
    "depending on whether it is 2D or 3D."
    assert 'image_dimensions' in context_config, "Context config must contain the key 'image_dimensions' to specify the dimensions of the image volume which the bbox is situated in."
    assert context_config['dimensionality']['expected_dimensionality'] in [2, 3], "Context config 'expected_dimensionality' must be either 2 or 3."
    if context_config['dimensionality']['expected_dimensionality'] == 2:
        assert 'collapsed_dimension' in context_config['dimensionality'], "Context config must contain the key 'collapsed_dimension' to specify which dimension is collapsed for 2D bounding box generation."
        assert context_config['dimensionality']['collapsed_dimension'] in [0, 1, 2], "Context config 'collapsed_dimension' must be 0, 1, or 2 to specify which dimension is collapsed for 2D bounding box generation."
    
    failure_bool = False
    invalid_dimensions = []
    #We start off with the assumption that the bbox is valid, and find evidence to the contrary! 

    min_x, min_y, min_z, max_x, max_y, max_z = bbox_extrema[0].tolist()
    
    #Check 1: min_x <= max_x, min_y <= max_y, min_z <= max_z
    if min_x > max_x or min_y > max_y or min_z > max_z:
        if critical_failure:
            raise ValueError("Invalid bbox: min coordinates must be less than or equal to max coordinates.")
        else:
            if min_x > max_x:
                invalid_dimensions.extend([0,3])
            if min_y > max_y:
                invalid_dimensions.extend([1,4])
            if min_z > max_z:
                invalid_dimensions.extend([2,5])
            
            failure_bool = True 
    else: 
        pass #Being explicit. ^
    
    #Check 2: All coordinates must be non-negative
    if min_x < 0 or min_y < 0 or min_z < 0:
        if critical_failure:
            raise ValueError("Invalid bbox: all coordinates must be non-negative.")
        else:
            if min_x < 0:
                invalid_dimensions.extend([0,3]) #NOTE: We return BOTH because otherwise, only reverting one value could result in another degenerate
                #bbox! We will always assume that for the case where this function is called after an augmentation, the original bbox was valid! So
                #we will safely revert both min and the max values.
            if min_y < 0:
                invalid_dimensions.extend([1,4])
            if min_z < 0:
                invalid_dimensions.extend([2,5])
            
            failure_bool = True
    #Check 3: All coordinates must be within the bounds of the image dimensions (on the upper end, we already checked
    #the lower end with the non-negativity check).
    
    #We can if max_x,y,z >= (EQUAL) the image dimension size as it is 0 indexed, and so the maximum valid coordinate
    #would be image_dim_size - 1.  
    if max_x >= context_config['image_dimensions'][0] or max_y >= context_config['image_dimensions'][1] or max_z >= context_config['image_dimensions'][2]:
        if critical_failure:
            raise ValueError("Invalid bbox: all coordinates must be within the bounds of the image dimensions.")
        else:
            if max_x >= context_config['image_dimensions'][0]:
                invalid_dimensions.extend([0,3]) #NOTE: We return BOTH because otherwise, only reverting one value could result in another degenerate
                #bbox! We will always assume that for the case where this function is called after an augmentation, the original bbox was valid! So
                #we will safely revert both min and the max values.
            if max_y >= context_config['image_dimensions'][1]:
                invalid_dimensions.extend([1,4])
            if max_z >= context_config['image_dimensions'][2]:
                invalid_dimensions.extend([2,5])
            
            failure_bool = True
    #Check 4: The bbox should be non-degenerate, and should match the expected dimensionality.

    #First check, is it a point? This is an obvious degenerate case.
    if min_x == max_x and min_y == max_y and min_z == max_z:
        if critical_failure:
            raise ValueError("Invalid bbox: bbox cannot be a point, it is degenerate.")
        else:
            invalid_dimensions.extend([0,1,2,3,4,5]) #All dimensions are invalid in this case.
            failure_bool = True
    else:
        #Now we check whether it matches the expected dimensionality.
        if context_config['dimensionality']['expected_dimensionality'] == 2:
            #We expect one collapsed dimension only.
            
            #We first find all of the dims where the min and max are the same, then we cross-reference that with the
            #expected collapsed dimension list. If there is a match, then all is good, if not then we have an invalid
            #bbox.

            collapsed_bbox_dims = [i for i in range(3) if bbox_extrema[0, i] == bbox_extrema[0, i + 3]]
            if collapsed_bbox_dims != [context_config['dimensionality']['collapsed_dimension']]:
                if critical_failure:
                    raise ValueError("Invalid bbox: for a 2D bbox, there must be exactly one collapsed dimension, and it must match the expected collapsed dimension specified in the context config.")
                else:
                    #We will mark all dimensions as invalid in the following way:
                    # 1) Does not match the expected collapsed dimension and is collapsed.
                    # 2) Matches the expected collapsed dimension but is not collapsed. 
                    for dim in range(3):
                        if dim != context_config['dimensionality']['collapsed_dimension'] and dim in collapsed_bbox_dims:
                            invalid_dimensions.extend([dim, dim + 3]) #Both min and max extrema for this dimension are invalid.
                        elif dim == context_config['dimensionality']['collapsed_dimension'] and dim not in collapsed_bbox_dims:
                            invalid_dimensions.extend([dim, dim + 3]) #Both min and max extrema for this dimension are invalid.
                    failure_bool = True
            else:
                pass #Being explicit. ^
        elif context_config['dimensionality']['expected_dimensionality'] == 3:
            #In this case, we expect no collapsed dimensions.
            collapsed_bbox_dims = [i for i in range(3) if bbox_extrema[0, i] == bbox_extrema[0, i + 3]]
            if len(collapsed_bbox_dims) > 0:
                if critical_failure:
                    raise ValueError("Invalid bbox: for a 3D bbox, there cannot be any collapsed dimensions, but the following dimensions are collapsed: {}".format(collapsed_bbox_dims))
                else:
                    #We will mark all dimensions as invalid which are collapsed. 
                    for dim in collapsed_bbox_dims:
                        invalid_dimensions.extend([dim, dim + 3]) #Both min and max extrema for this dimension are invalid.
                    failure_bool = True
        else:
            raise ValueError("Unsupported dimensionality in context config. Dimensionality must be either 2 or 3.")

    #Now we will reduce the invalid dimensions to non-redundant entries. 
    invalid_dimensions = list(set(invalid_dimensions))
    assert failure_bool == bool(invalid_dimensions), "Failure bool should be True if and only if there are invalid dimensions, i.e., the list of invalid dimensions is not empty."
    assert len(invalid_dimensions) % 2 == 0, "Invalid dimensions should come in pairs of min and max for the same dimension, so the length of the invalid dimensions"
    "list should be even."

    for dim in range(3):
        if dim in invalid_dimensions and (dim + 3) not in invalid_dimensions:
            raise ValueError("Invalid dimensions should come in pairs of min and max for the same dimension, but dimension {} is marked as invalid without its pair.".format(dim))
        if (dim + 3) in invalid_dimensions and dim not in invalid_dimensions:
            raise ValueError("Invalid dimensions should come in pairs of min and max for the same dimension, but dimension {} is marked as invalid without its pair.".format(dim + 3))
        
    return not failure_bool, invalid_dimensions

