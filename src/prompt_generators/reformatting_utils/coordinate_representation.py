import torch
import numpy as np
import os
from os.path import dirname as up
import sys
sys.path.append(up(up(up(up(os.path.abspath(__file__))))))


def voxel_to_coordinate_conversion(
    voxel_coords: torch.Tensor,
    image_shape: torch.Size = None
) -> torch.Tensor:
    '''
    Function which converts a voxel coordinate (zero-indexed, grid/cell level) to a coordinate 
    based on the corner of the voxel being 0,0,0.
    
    This converts voxel indices to continuous coordinates where each voxel is centered at its position.
    For a voxel at index [i, j, k], the corresponding coordinate is [i+0.5, j+0.5, k+0.5].
    
    Args:
        voxel_coords: A tensor of shape (N, 3) or (3,) containing the voxel indices [i, j, k].
            Can be a single triplet [i, j, k] or multiple triplets.
        image_shape: Optional torch.Size or tuple representing the image dimensions (D, H, W).
            If provided, the function will validate that the voxel coordinates are within bounds.
    
    Returns:
        A tensor of the same shape as voxel_coords, where each voxel index has been converted
        to its corresponding continuous coordinate by adding 0.5 to each dimension.
        
    Examples:
        >>> voxel_to_coordinate_conversion(torch.tensor([0, 0, 0]))
        tensor([0.5000, 0.5000, 0.5000])
        
        >>> voxel_to_coordinate_conversion(torch.tensor([[0, 0, 0], [5, 10, 15]]))
        tensor([[0.5000, 0.5000, 0.5000],
                [5.5000, 10.5000, 15.5000]])
                
        >>> # With image shape validation
        >>> voxel_to_coordinate_conversion(
        ...     torch.tensor([0, 0, 0]), 
        ...     image_shape=(64, 64, 64)
        ... )
        tensor([0.5000, 0.5000, 0.5000])
        
        >>> # This would raise an error if out of bounds
        >>> voxel_to_coordinate_conversion(
        ...     torch.tensor([64, 64, 64]), 
        ...     image_shape=(64, 64, 64)
        ... )  # 64 is out of bounds for a 64x64x64 image (valid: 0-63)
    '''
    # Ensure input is a tensor
    if not isinstance(voxel_coords, torch.Tensor):
        voxel_coords = torch.tensor(voxel_coords)
    
    # Handle both single triplet and batch of triplets
    if voxel_coords.dim() == 1:
        # Single triplet - reshape to (1, 3) for processing
        original_shape = voxel_coords.shape
        voxel_coords = voxel_coords.unsqueeze(0)
        single_input = True
    else:
        single_input = False
        original_shape = None
    
    # Validate input shape
    if voxel_coords.shape[1] != 3:
        raise ValueError(f"Input must have shape (N, 3) or (3,), got {voxel_coords.shape}")
    
    # Validate image bounds if provided
    if image_shape is not None:
        if len(image_shape) != 3:
            raise ValueError(f"Image shape must have 3 dimensions, got {len(image_shape)}")
        
        # Check that all coordinates are within bounds
        # Valid indices are [0, image_dim-1] for each dimension
        max_coords = voxel_coords.max(dim=0)[0]
        for i, dim_size in enumerate(image_shape):
            if max_coords[i] > dim_size - 1:
                raise ValueError(
                    f"Voxel coordinate {max_coords[i]} in dimension {i} exceeds image bounds "
                    f"(image size: {dim_size}, valid indices: 0 to {dim_size-1})"
                )
            if voxel_coords[:, i].min() < 0:
                raise ValueError(
                    f"Voxel coordinate in dimension {i} is negative. "
                    f"Minimum value: {voxel_coords[:, i].min()}"
                )
    
    # Validate that voxel coordinates are integers
    # Voxel indices must be integer values (though not necessarily int dtype)
    if not torch.allclose(voxel_coords, torch.round(voxel_coords)):
        raise ValueError("Voxel coordinates must be integer values. Got non-integer values.")
    
    # Convert voxel indices to continuous coordinates
    # Each voxel at index i occupies the space [i, i+1], so its center is at i + 0.5
    converted_coords = voxel_coords.float() + 0.5
    
    # Restore original shape if input was a single triplet
    if single_input:
        converted_coords = converted_coords.squeeze(0)
    
    return converted_coords


def coordinate_to_voxel_conversion(
    coords: torch.Tensor,
    image_shape: torch.Size = None
) -> torch.Tensor:
    '''
    Function which converts continuous coordinates back to voxel indices.
    
    For any coordinate in the half-open interval [i, i+1), this returns i.
    This is the inverse of voxel_to_coordinate_conversion only when the
    coordinate is at the voxel center (i + 0.5).
    
    Args:
        coords: A tensor of shape (N, 3) or (3,) containing continuous coordinates.
        image_shape: Optional torch.Size or tuple representing the image dimensions (D, H, W).
            If provided, the function will validate that the coordinates are within bounds.
    
    Returns:
        A tensor of the same shape as coords, where each continuous coordinate has been 
        converted to its corresponding voxel index by flooring.
        
    Examples:
        >>> coordinate_to_voxel_conversion(torch.tensor([0.5, 0.5, 0.5]))
        tensor([0, 0, 0])
        
        >>> coordinate_to_voxel_conversion(torch.tensor([5.5, 10.5, 15.5]))
        tensor([5, 10, 15])
    '''
    # Ensure input is a tensor
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords)
    
    # Handle both single triplet and batch of triplets
    if coords.dim() == 1:
        original_shape = coords.shape
        coords = coords.unsqueeze(0)
        single_input = True
    else:
        single_input = False
        original_shape = None
    
    # Validate input shape
    if coords.shape[1] != 3:
        raise ValueError(f"Input must have shape (N, 3) or (3,), got {coords.shape}")
    
    # Validate image bounds if provided
    if image_shape is not None:
        if len(image_shape) != 3:
            raise ValueError(f"Image shape must have 3 dimensions, got {len(image_shape)}")
        
        # Check that all coordinates are within bounds
        # Valid continuous coordinates are [0, image_dim) for each dimension
        # A coordinate of dim_size is at the boundary and outside the valid image space
        max_coords = coords.max(dim=0)[0]
        for i, dim_size in enumerate(image_shape):
            if max_coords[i] >= dim_size:
                raise ValueError(
                    f"Coordinate {max_coords[i]} in dimension {i} exceeds image bounds "
                    f"(image size: {dim_size}, valid range: 0 to {dim_size-1})"
                )
            if coords[:, i].min() < 0:
                raise ValueError(
                    f"Coordinate in dimension {i} is negative. "
                    f"Minimum value: {coords[:, i].min()}"
                )
    
    # Convert continuous coordinates to voxel indices
    # floor gives the voxel index for any coordinate in [i, i+1)
    voxel_indices = torch.floor(coords.float()).long()
    
    # Restore original shape if input was a single triplet
    if single_input:
        voxel_indices = voxel_indices.squeeze(0)
    
    return voxel_indices


def voxel_to_coordinate_conversion_bbox(
    bbox_coords: torch.Tensor,
    image_shape: torch.Size = None
) -> torch.Tensor:
    '''
    Wrapper around voxel_to_coordinate_conversion for bbox extrema tensors of shape (1, 6).
    Reshapes (1, 6) into (2, 3) [min extrema, max extrema], applies the standard conversion,
    then reshapes back to (1, 6).
    '''
    coords_2x3 = bbox_coords.reshape(2, 3)
    converted = voxel_to_coordinate_conversion(coords_2x3, image_shape)
    return converted.reshape(1, 6)


def coordinate_to_voxel_conversion_bbox(
    coords: torch.Tensor,
    image_shape: torch.Size = None
) -> torch.Tensor:
    '''
    Wrapper around coordinate_to_voxel_conversion for bbox extrema tensors of shape (1, 6).
    Reshapes (1, 6) into (2, 3) [min extrema, max extrema], applies the standard conversion,
    then reshapes back to (1, 6).
    '''
    coords_2x3 = coords.reshape(2, 3)
    converted = coordinate_to_voxel_conversion(coords_2x3, image_shape)
    return converted.reshape(1, 6)


if __name__ == '__main__':
    # Test cases
    print("Testing voxel_to_coordinate_conversion:")
    
    # Test 1: Single voxel at origin
    result1 = voxel_to_coordinate_conversion(torch.tensor([0, 0, 0]))
    print(f"  [0, 0, 0] -> {result1}")
    assert torch.allclose(result1, torch.tensor([0.5, 0.5, 0.5]))
    
    # Test 2: Multiple voxels
    result2 = voxel_to_coordinate_conversion(torch.tensor([[0, 0, 0], [5, 10, 15]]))
    print(f"  [[0,0,0], [5,10,15]] -> {result2}")
    expected2 = torch.tensor([[0.5, 0.5, 0.5], [5.5, 10.5, 15.5]])
    assert torch.allclose(result2, expected2)
    
    # Test 3: Voxel near edge
    result3 = voxel_to_coordinate_conversion(torch.tensor([63, 63, 63]))
    print(f"  [63, 63, 63] -> {result3}")
    assert torch.allclose(result3, torch.tensor([63.5, 63.5, 63.5]))
    
    # Test 4: With image shape validation
    result4 = voxel_to_coordinate_conversion(
        torch.tensor([0, 0, 0]), 
        image_shape=(64, 64, 64)
    )
    print(f"  [0, 0, 0] with shape (64,64,64) -> {result4}")
    
    print("\nTesting coordinate_to_voxel_conversion:")
    
    # Test 5: Convert back
    result5 = coordinate_to_voxel_conversion(torch.tensor([0.5, 0.5, 0.5]))
    print(f"  [0.5, 0.5, 0.5] -> {result5}")
    assert torch.allclose(result5, torch.tensor([0, 0, 0]))
    
    # Test 6: Multiple coordinates
    result6 = coordinate_to_voxel_conversion(torch.tensor([[0.5, 0.5, 0.5], [5.5, 10.5, 15.5]]))
    print(f"  [[0.5,0.5,0.5], [5.5,10.5,15.5]] -> {result6}")
    expected6 = torch.tensor([[0, 0, 0], [5, 10, 15]])
    assert torch.allclose(result6, expected6)
    
    print("\nAll tests passed!")