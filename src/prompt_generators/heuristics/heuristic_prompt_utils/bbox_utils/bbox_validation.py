"""
Validation functions for checking if a binary mask can generate a valid bbox.
These functions use a fast pre-check based on contiguous sequence analysis
before proceeding to connected component analysis/or as a post-component extraction step to verify if the 
components constitute those which can be used to generate a valid bbox.

The main idea is to check if there are at least N contiguous coords in each relevant dimension (x, y, and z for 3D) where N is the minimum length required for a valid bbox (e.g., 2).

"""
import torch
from typing import Optional, Tuple

def has_contiguous_sequence_vectorised(unique_coords, min_length=2):
    """
    Vectorised process for checking contiguous sequences in multiple dimensions
    (one per dimension) simultaneously using a padded tensor.

    Args:
        unique_coords: 2D tensor of shape (num_dimensions, max_coords) where:
            - Each row represents unique coordinates for one dimension
            - Shorter coordinate arrays are padded with torch.nan
        min_length: Minimum length of contiguous sequence required (default: 2), translates to min_length - 1 in actual distance
        between the centres which are indicating the bbox extrema. 
    
    Returns:
        True if ALL dimensions have a contiguous sequence of at least min_length,
        False otherwise.
    """
    num_dims, max_coords = unique_coords.shape
    
    # Step 1: Early exit if not enough coordinates, we need at least min_length unique coordinates. E.g. 0,1 would be length 2
    # in voxel count covered. 
    if max_coords < min_length:
        return False
    
    # Step 1b: Check that each dimension has at least min_length valid coordinates
    valid_mask = ~torch.isnan(unique_coords)
    valid_counts = valid_mask.sum(dim=1)
    if (valid_counts < min_length).any():
        return False
    
    # Step 2: Create valid mask and replace padded values with NaN
    # This ensures padded values don't affect the diff calculation
    coords_with_nan = torch.where(valid_mask, unique_coords.float(), torch.nan)
    
    # Step 3: Sort each row (NaN values will be at the end by default)
    sorted_coords, _ = torch.sort(coords_with_nan, dim=1)
    
    # Step 4: Compute differences between consecutive coordinates
    # Each diff represents the gap between adjacent coordinates
    diffs = torch.diff(sorted_coords, dim=1)
    
    # Step 5: Identify contiguous sequences (diff == 1)
    # A diff of 1 means two consecutive coordinates are adjacent
    is_contiguous = (diffs == 1)
    
    # Step 6: Set up sliding window parameters
    window_size = min_length - 1  # Need min_length-1 consecutive diffs of 1
    diff_len = diffs.size(1)      # Number of diff values
    
    # Early exit if not enough diffs for a window
    if diff_len < window_size:
        return False
    
    # Step 7: Create sliding window indices
    # num_windows = number of possible starting positions for a window
    num_windows = diff_len - window_size + 1
    
    # window_indices shape: (num_windows, window_size)
    # Each row contains the indices for one sliding window
    # Example: if window_size=3, diff_len=5:
    #   row 0: [0, 1, 2]  - window starting at position 0
    #   row 1: [1, 2, 3]  - window starting at position 1
    #   row 2: [2, 3, 4]  - window starting at position 2
    window_indices = torch.arange(num_windows, device=unique_coords.device).unsqueeze(1) + \
                     torch.arange(window_size, device=unique_coords.device).unsqueeze(0)
    
    # Step 8: Expand is_contiguous for batch processing
    # is_contiguous shape: (num_dims, diff_len)
    # After unsqueeze: (num_dims, 1, diff_len)
    # After expand: (num_dims, num_windows, diff_len)
    # Each "layer" along dimension 1 is a copy of the original is_contiguous row
    is_contiguous_expanded = is_contiguous.unsqueeze(1).expand(num_dims, num_windows, diff_len)
    
    # Step 9: Expand window_indices for batch processing
    # window_indices shape: (num_windows, window_size)
    # After unsqueeze: (1, num_windows, window_size)
    # After expand: (num_dims, num_windows, window_size)
    # Each batch gets the same window indices
    window_indices_expanded = window_indices.unsqueeze(0).expand(num_dims, -1, -1)
    
    # Step 10: Gather values using advanced indexing, we essentially march the sliding window across the is_contiguous array
    #by using the window indices. This is why we needed to expand both the is_contiguous and the window_indices tensors to have a batch dimension for the dimensions.
    
    # For each dimension d, window w, and position p in the window:
    #   windows[d, w, p] = is_contiguous_expanded[d, w, window_indices[d, w, p]]
    # This extracts the sliding windows from each dimension's is_contiguous array
    # Result shape: (num_dims, num_windows, window_size)
    windows = is_contiguous_expanded.gather(2, window_indices_expanded)
    
    # Step 11: Sum across window dimension
    # Each window sum represents the count of contiguous diffs in that window
    # Result shape: (num_dims, num_windows)
    window_sums = windows.sum(dim=-1)
    
    # Step 12: Check if any window has all contiguous diffs
    # A window with sum == window_size means all values are True (contiguous)
    # any(dim=1): Check if any window in each dimension has a contiguous sequence
    # all(): Check if ALL dimensions have at least one contiguous sequence
    has_contiguous = ((window_sums == window_size).any(dim=1)).all()
    if has_contiguous:
        return True
    else:        
        return False

def can_generate_bbox_from_slice_fast(slice_2d: torch.Tensor) -> bool:
    """
    Fast pre-check: Can this 2D slice potentially generate a valid bbox?
    
    A valid bbox requires:
    - At least 2 foreground voxels
    - A contiguous sequence of at least 2 in the x dimension
    - A contiguous sequence of at least 2 in the y dimension
    
    This is a fast check that uses contiguous sequence analysis. can be used before a more expensive connected 
    component analysis, or to verify whether a bbox can be generated from extracted connected components.
    
    Args:
        slice_2d: 2D binary mask (H, W)
    
    Returns:
        True if the slice can potentially generate a bbox, False otherwise
    """
    foreground = torch.nonzero(slice_2d, as_tuple=False)
    
    if foreground.shape[0] < 2: #Trivial check for a single point.
        return False
    
    unique_x = torch.unique(foreground[:, 0])
    unique_y = torch.unique(foreground[:, 1])
    
    # Pad the smaller array with NaNs to match lengths
    max_len = max(len(unique_x), len(unique_y))
    
    # Pad unique_x to max_len with NaNs
    if len(unique_x) < max_len:
        padding_x = torch.full((max_len - len(unique_x),), torch.nan, 
                               device=unique_x.device)
        unique_x = torch.cat([unique_x, padding_x])
    
    # Pad unique_y to max_len with NaNs
    if len(unique_y) < max_len:
        padding_y = torch.full((max_len - len(unique_y),), torch.nan, 
                                 device=unique_y.device)
        unique_y = torch.cat([unique_y, padding_y])
    
    # Stack into a 2D tensor (2, max_len)
    unique_coords = torch.stack([unique_x, unique_y])
    
    return has_contiguous_sequence_vectorised(unique_coords, 2)

def can_generate_bbox_from_volume_fast(volume_3d: torch.Tensor) -> bool:
    """
    Fast pre-check: Can this 3D volume potentially generate a valid bbox?
    
    A valid bbox requires:
    - At least 2 foreground voxels
    - A contiguous sequence of at least 2 in the x dimension
    - A contiguous sequence of at least 2 in the y dimension
    - A contiguous sequence of at least 2 in the z dimension
    
    This is a fast check that uses contiguous sequence analysis. can be used before a more expensive connected 
    component analysis, or to verify whether a bbox can be generated from extracted connected components.
    
    Args:
        volume_3d: 3D binary mask (D, H, W)
    
    Returns:
        True if the volume can potentially generate a bbox, False otherwise
    """
    foreground = torch.nonzero(volume_3d, as_tuple=False)
    
    if foreground.shape[0] < 2:
        return False
    
    unique_x = torch.unique(foreground[:, 0])
    unique_y = torch.unique(foreground[:, 1])
    unique_z = torch.unique(foreground[:, 2])
    
    # Pad all arrays with NaNs to match lengths
    max_len = max(len(unique_x), len(unique_y), len(unique_z))
    
    # Pad unique_x to max_len with NaNs
    if len(unique_x) < max_len:
        padding_x = torch.full((max_len - len(unique_x),), torch.nan, device=unique_x.device)
        unique_x = torch.cat([unique_x, padding_x])
    
    # Pad unique_y to max_len with NaNs
    if len(unique_y) < max_len:
        padding_y = torch.full((max_len - len(unique_y),), torch.nan, device=unique_y.device)
        unique_y = torch.cat([unique_y, padding_y])
    
    # Pad unique_z to max_len with NaNs
    if len(unique_z) < max_len:
        padding_z = torch.full((max_len - len(unique_z),), torch.nan, device=unique_z.device)
        unique_z = torch.cat([unique_z, padding_z])
    
    # Stack into a 2D tensor (3, max_len)
    unique_coords = torch.stack([unique_x, unique_y, unique_z])
    
    return has_contiguous_sequence_vectorised(unique_coords, 2)

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

