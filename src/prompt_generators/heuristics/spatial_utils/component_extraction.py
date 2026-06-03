import torch
import numpy as np
import warnings
from typing import Any, List, Optional, Tuple
from skimage import measure as measure_np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from src.prompt_generators.heuristics.heuristic_prompt_utils.bbox_utils.bbox_validation import (
    can_generate_bbox_from_slice_fast,
    can_generate_bbox_from_volume_fast
)
from src.version_handling import monai_version


def convert_to_numpy(mask: Any) -> np.ndarray:
    """
    Convert a tensor or MetaTensor to numpy array.
    
    For MetaTensor (MONAI 1.4+), uses .array property.
    For regular tensors, uses .numpy() method.
    Automatically moves CUDA tensors to CPU first.
    
    Args:
        mask: A tensor or MetaTensor object
        
    Returns:
        numpy.ndarray: The converted numpy array
    """
    if monai_version == '1.4.0':
        # MONAI 1.4+ MetaTensor has .array property
        if hasattr(mask, 'array'):
            arr = mask.array
            if isinstance(arr, torch.Tensor):
                if arr.device.type == 'cuda':
                    arr = arr.cpu()
                return arr.numpy()
            return arr
        elif hasattr(mask, 'numpy'):
            if mask.device.type == 'cuda':
                mask = mask.cpu()
            return mask.numpy()
        else:
            raise ValueError(f"Cannot convert object of type {type(mask)} to numpy array")
    elif monai_version == '0.9.0':
        # MONAI 0.9.0 uses .data for MetaTensor
        if hasattr(mask, 'data'):
            data = mask.data
            if hasattr(data, 'numpy'):
                if data.device.type == 'cuda':
                    data = data.cpu()
                return data.numpy()
            else:
                return data
        elif hasattr(mask, 'numpy'):
            if mask.device.type == 'cuda':
                mask = mask.cpu()
            return mask.numpy()
        else:
            raise ValueError(f"Cannot convert object of type {type(mask)} to numpy array")
    else:
        raise ValueError(f"Unsupported MONAI version: {monai_version}")


def validate_connectivity(orthogonal_hops: int, ndim: int) -> int:
    """
    Validate and convert orthogonal hops to skimage connectivity parameter.
    
    The connectivity is expressed as the number of orthogonal hops:
    - 1 hop: Only face-adjacent neighbors (4-connectivity in 2D, 6-connectivity in 3D)
    - 2 hops: Face + edge-adjacent neighbors (8-connectivity in 2D, 18-connectivity in 3D)
    - 3 hops: All neighbors including corners (8-connectivity in 2D, 26-connectivity in 3D)
    
    Args:
        orthogonal_hops: Number of orthogonal hops (1, 2, or 3). Must be <= ndim.
        ndim: The dimensionality of the image (2 or 3)
        
    Returns:
        The skimage connectivity parameter (1, 2, or 3)
        
    Raises:
        ValueError: If orthogonal_hops is not valid for the given dimensionality
    """
    if orthogonal_hops < 1:
        raise ValueError(f"Orthogonal hops must be at least 1, got {orthogonal_hops}")
    
    if orthogonal_hops > ndim:
        raise ValueError(
            f"Orthogonal hops {orthogonal_hops} is not valid for {ndim}D images. "
            f"Maximum allowed is {ndim}."
        )
    
    # Map orthogonal hops directly to skimage connectivity parameter
    return orthogonal_hops


def extract_connected_components(
    mask: torch.Tensor,
    orthogonal_hops: int = 1,
    device: str = None
) -> torch.Tensor:
    """
    Extract connected components from a binary mask.
    
    Args:
        mask: A binary mask (torch tensor, possibly on CUDA device).
              Can be 2D (H, W) or 3D (H, W, D).
        orthogonal_hops: Number of orthogonal hops for connectivity:
                        - 1: Only face-adjacent neighbors (4-connectivity in 2D, 6-connectivity in 3D)
                        - 2: Face + edge-adjacent neighbors (8-connectivity in 2D, 18-connectivity in 3D)
                        - 3: All neighbors including corners (26-connectivity in 3D)
                        Must be <= ndim.
        device: The device to place the output components on. If None, uses the input mask's device.
                Can be a string like 'cpu', 'cuda', or a torch.device object.
                      
    Returns:
        torch.Tensor: A single labeled mask where each connected component has a unique integer ID.
        
    Raises:
        ValueError: If the mask is not 2D or 3D, or if orthogonal_hops is invalid for the dimensionality.
    """
    # Validate input dimensions
    if mask.dim() not in (2, 3):
        raise ValueError(f"Only 2D and 3D masks are supported, got {mask.dim()}D")
    
    # Validate and convert orthogonal hops to skimage connectivity
    sk_connectivity = validate_connectivity(orthogonal_hops, mask.dim())
    # Convert to numpy array (handles both MetaTensor and regular tensors)
    mask_np = convert_to_numpy(mask)
    
    # Ensure binary format
    mask_np = (mask_np > 0).astype(np.uint8)
    
    # Label connected components using skimage.measure.label
    labeled, num_features = measure_np.label(mask_np, connectivity=sk_connectivity, return_num=True)
    
    if num_features == 0:
        raise ValueError("No connected components found in the mask.")
    
    # Determine the target device
    if device is None:
        target_device = mask.device
    else:
        target_device = device
    
    # Convert labeled mask back to torch tensor on the specified device
    # Use int32 as it's sufficient for most use cases (supports up to 2^31-1 components)
    labeled_tensor = torch.from_numpy(labeled).to(device=target_device, dtype=torch.int32)


    #Now we will need to filter out incompatible components.


    torch.cuda.empty_cache()  # Clear CUDA cache to free up memory after conversion
    return labeled_tensor


def two_d_components_generation(
    binary_mask: torch.Tensor,
    slice_selection_config: dict,
    connectivity: int
):
    '''
    Helper function which selects a slice and extracts connected components from a binary mask based on a 
    specified strategy, this is used for 2D bounding box generation.

    Given that 

    inputs:
    binary_mask: A binary mask tensor of shape (H, W, D)
    slice_selection_config: A dictionary containing the selection strategy and any required parameters, expected structure:
    {
        'slice_selection_strategy': str,  # 'center', 'top', 'bottom', 'random'
        'collapsed_dim': int,  # 0, 1, or 2 - which dimension is collapsed for 2D bounding box generation -> i.e., which dimension to
                            # select the slice from.
    }  
    connectivity: int, the connectivity to use for connected component analysis (1, 2)
    outputs:
    selected_slice_idx: int, the index of the selected slice along the collapsed dimension.
    '''
    assert binary_mask.dim() == 3, "Input binary mask must be 3D for slice selection."
    if any(binary_mask.shape[i] == 0 for i in range(3)):
        raise ValueError("Input binary mask must have non-zero size in all dimensions.")
    if connectivity not in (1, 2):
        raise ValueError("Connectivity must be 1 or 2 for 2D masks.")
    
    dims_to_sum = [i for i in range(3) if i != slice_selection_config['collapsed_dim']]
    non_zero_slices = torch.nonzero(torch.sum(binary_mask, dim=dims_to_sum) > 0, as_tuple=False)
    if len(non_zero_slices) == 0:
            raise ValueError("No non-zero slices found in binary mask.")
    #assert that the non zero slices are not repeated.
    if torch.unique(non_zero_slices, return_counts=True)[1].max() > 1:
        raise ValueError("Non-zero slices are repeated, which should not happen.")
    
    candidate_region = None
    slice_idx = None
    
    while non_zero_slices.shape[0] > 0:
        # Determine which slice to use based on slice_selection
        if slice_selection_config['slice_selection_strategy'] == 'center':
            # Find the center slice with non-zero values    
            slice_idx = non_zero_slices[len(non_zero_slices) // 2].item()
        elif slice_selection_config['slice_selection_strategy'] == 'top':
            # Find the topmost (first) slice with non-zero values
            slice_idx = non_zero_slices[0].item()
        elif slice_selection_config['slice_selection_strategy'] == 'bottom':
            # Find the bottommost (last) slice with non-zero values
            slice_idx = non_zero_slices[-1].item()
        elif slice_selection_config['slice_selection_strategy'] == 'random':
            # Randomly select a slice with non-zero values
            slice_idx = non_zero_slices[torch.randint(len(non_zero_slices), (1,)).item()].item()
        
        # Extract the 2D slice
        candidate_region = binary_mask[tuple(slice_idx if i == slice_selection_config['collapsed_dim'] else slice(None) for i in range(3))]

        # #We run the fast-check for compatibility, if it fails then we will remove this slice and move onto the 
        # #next one.
        # can_generate_bbox, _ = fast_check_bbox_generation_compatibility(candidate_region, slice_selection_config['collapsed_dim'])

        is_compatible, components = generate_components_from_mask(candidate_region, 2, connectivity)
        if is_compatible:
            assert components.sum() > 0, "Candidate region should contain non-zero values if it is compatible."
            #If compatible then return slice_idx and the candidate region.
            return components, slice_idx, is_compatible
        # Remove the selected slice from non_zero_slices for next iteration
        non_zero_slices = non_zero_slices[non_zero_slices != slice_idx]

    #NOTE: If we have reached here then we have exited the loop without finding a compatible slice, so now
    # we will need to return an empty tensor with a None slice_idx.
    return torch.zeros_like(binary_mask[tuple(slice(None) if i != slice_selection_config['collapsed_dim'] else 0 for i in range(3))]), None, False


def three_d_components_generation(
    binary_mask: torch.Tensor,
    connectivity: int
):
    '''
    Helper function which selects a slice and extracts connected components from a binary mask based on a 
    specified strategy, this is used for 3D bounding box generation.

    Given that 

    inputs:
    binary_mask: A binary mask tensor of shape (H, W, D)
    connectivity: int, the connectivity to use for connected component analysis (1, 2, 3)

    outputs:
    selected_slice_idx: int, the index of the selected slice along the collapsed dimension.
    '''
    assert binary_mask.dim() == 3, "Input binary mask must be 3D."
    if any(binary_mask.shape[i] == 0 or binary_mask.shape[i] < 2 for i in range(3)):
        raise ValueError("Input binary mask must have non-zero size and at least 2 in all dimensions.")
    if connectivity not in (1, 2, 3):
        raise ValueError("Connectivity must be 1, 2, or 3 for 3D masks.")
    
    if binary_mask.sum() == 0 and binary_mask.unique() == torch.tensor([0], device=binary_mask.device):
        raise ValueError("Input binary mask must contain non-zero values.")
    
    is_compatible, components = generate_components_from_mask(binary_mask, 3, connectivity)
    return components, is_compatible 


##################################################################################################################

def generate_components_from_mask(
    binary_mask: torch.Tensor,
    dimensionality: int,
    connectivity: int = 1
) -> Tuple[bool, Optional[torch.Tensor]]:
    """
    Check if a binary mask can potentially generate a valid bbox.
    
    This function performs:
    1. Fast pre-check using contiguous sequence analysis
    2. If that passes, runs connected component analysis
    3. Filters components to ensure each has length >= 2 in all dimensions
    
    Args:
        binary_mask: Binary mask tensor (2D or 3D)
        dimensionality: 2 or 3, indicating the dimensionality of the mask
        connectivity: Connectivity for connected component analysis (1, 2, or 3)
    
    Returns:
        Tuple of (is_compatible, components) where:
            - is_compatible: True if the mask can generate a bbox with valid components
            - components: The connected components tensor if valid, None otherwise
    """
    assert binary_mask.ndim == dimensionality, f"Expected binary mask with {dimensionality}" 
    f"dimensions, got {binary_mask.ndim}"
    if dimensionality == 2:
        # Fast pre-check for 2D
        if not can_generate_bbox_from_slice_fast(binary_mask):
            return False, None
        
        try:
            components = extract_connected_components(binary_mask, connectivity)
            if components.max() == 0:
                return False, None
            
            # Filter components: each must have length >= 2 in all dimensions
            valid_components = filter_valid_components(components, dimensionality)
            if valid_components.max() == 0:
                return False, None
            return True, valid_components
        except ValueError:
            return False, None
    
    elif dimensionality == 3:
        # Fast pre-check for 3D
        if not can_generate_bbox_from_volume_fast(binary_mask):
            return False, None
        
        try:
            components = extract_connected_components(binary_mask, connectivity)
            if components.max() == 0:
                return False, None
            
            # Filter components: each must have length >= 2 in all dimensions
            valid_components = filter_valid_components(components, dimensionality)
            if valid_components.max() == 0:
                return False, None
            return True, valid_components
        except ValueError:
            return False, None
    
    else:
        raise ValueError(f"Unsupported dimensionality: {dimensionality}. Must be 2 or 3.")


def filter_valid_components(components: torch.Tensor, dimensionality: int) -> torch.Tensor:
    """
    Filter connected components to keep only those with length >= 2 in all dimensions.
    Uses the existing fast check functions to validate each component for this purpose.
    
    Args:
        components: Connected components tensor where each component has a unique integer ID
        dimensionality: 2 or 3, indicating the dimensionality of the mask
    
    Returns:
        A new components tensor with only valid components (0 for invalid/filtered out)
    """
    assert components.dim() == dimensionality, f"Expected components with {dimensionality} dimensions, got {components.dim()}"
    
    # Collect valid component IDs
    valid_ids = []
    for component_id in torch.unique(components):
        if component_id == 0:
            continue
        
        # Extract this component as a binary mask
        component_mask = (components == component_id)
        
        # Use the appropriate fast check function
        if dimensionality == 2:
            is_compatible = can_generate_bbox_from_slice_fast(component_mask)
        else:
            is_compatible = can_generate_bbox_from_volume_fast(component_mask)
        
        if is_compatible:
            valid_ids.append(component_id)
    
    # If no valid components, return zeros
    if len(valid_ids) == 0:
        return torch.zeros_like(components)
    
    # Create mapping from old ID to new sequential ID
    valid_ids = torch.tensor(valid_ids, device=components.device, dtype=torch.long)
    id_mapping = torch.zeros(components.max() + 1, dtype=torch.long, device=components.device)
    for new_id, old_id in enumerate(valid_ids, start=1):
        id_mapping[old_id] = new_id
    
    # Apply mapping to get new component IDs
    return id_mapping[components]


###########################################################################################################

def select_component(
    components: torch.Tensor,
    component_selection_config: dict,
    ):
    '''
    Function which selects connected components from a tensor which has different integer values representing different connected
    components, according to a specified selection process. 

    inputs:
    components: A tensor of shape (H,W) or (H,W,D) where the value of each voxel indicates the connected component it belongs to, with
    0 typically representing the background.
    component_selection_config: A dictionary containing the selection process and any required parameters, expected structure:
    {
    'component_selection_process': str,  # e.g., 'top-k'
    '{process}': Any,
    }

    outputs:
    selected_components: A tensor of the same shape as input, where only the selected components are retained, and converted to a 
    binary mask (i.e., values of 1 for selected components, 0 for background and non-selected components).
    '''
    if components.sum() == 0:
        raise ValueError("Input components tensor contains no non-zero values, cannot perform component selection.")
    if 'component_selection_process' not in component_selection_config:
        raise KeyError("component_selection_config must contain the key 'component_selection_process' to specify the selection process.")
    if component_selection_config['component_selection_process'] not in ['top-k']:
        raise ValueError(f"Invalid component_selection_process value: {component_selection_config['component_selection_process']}. "
                         f"Currently, only 'top-k' selection process is supported.")
    if component_selection_config['component_selection_process'] not in component_selection_config.keys():
        raise KeyError(f"The parameterisation of the component selection process must be specified in the component_selection_config under the key outlined by the selection process, i.e., {component_selection_config['component_selection_process']}")
    
    if component_selection_config['component_selection_process'] == 'top-k':
        assert component_selection_config['top-k'] > 0, "top-k parameter must be greater than 0 for top-k component selection."
        assert type(component_selection_config['top-k']) == int, "top-k parameter must be an integer for top-k component selection."
        
        # Sort components by size (descending) and select top-k
        #We get a list of the components by size, we do this by iterating on the unique ids (except 0) and summing the voxels.
        component_sizes = [(torch.sum(components == id), id) for id in torch.unique(components) if id != 0]
        components_sorted = [id for size, id in sorted(component_sizes, key=lambda x: x[0], reverse=True)]
        top_k_components = components_sorted[:component_selection_config['top-k']] if len(components_sorted) >= component_selection_config['top-k'] else components_sorted
        # Create binary mask containing all selected components using broadcasting:
        # - components.unsqueeze(0) adds a batch dimension to enable batch-wise comparison
        # - top-k_components reshaped to (K, 1, 1) for 2D or (K, 1, 1, 1) for 3D to broadcast across spatial dims
        # - The comparison produces an expanded boolean array where each "layer" corresponds to one selected component ID
        # - .any(dim=0) combines all layers with OR to produce the final binary mask
        top_k_tensor = torch.tensor(top_k_components)
        top_k_tensor = top_k_tensor.view((len(top_k_components),) + (1,) * components.dim())
        selected_components = (components.unsqueeze(0) == top_k_tensor).any(dim=0).float()

    return selected_components

if __name__ == '__main__':
    # Test ID mapping re-indexing
    print("=" * 50)
    print("Test: ID mapping re-indexing")
    print("=" * 50)
    
    input_mask = torch.tensor([
        [0, 1, 1, 1, 1, 0, 0, 2],
        [2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 1, 0, 2]
    ])
    
    # Simulate filtering: only component 1 is valid, component 2 is invalid
    valid_ids = torch.tensor([1], dtype=torch.long)
    id_mapping = torch.zeros(input_mask.max() + 1, dtype=torch.long)
    for new_id, old_id in enumerate(valid_ids, start=1):
        id_mapping[old_id] = new_id
    
    result = id_mapping[input_mask]
    expected = torch.tensor([
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0]
    ])
    assert torch.equal(result, expected), f"Expected:\n{expected}\nGot:\n{result}"
    print(f"Input mask:\n{input_mask}")
    print(f"Result after re-indexing:\n{result}")
    print("Test passed!")
    print()

    # Test 2D case
    binary_mask_2d = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.int64)

    labeled_mask = extract_connected_components(binary_mask_2d, orthogonal_hops=1)
    print(f"2D labeled mask shape: {labeled_mask.shape}")
    print(f"Number of components: {labeled_mask.max().item()}")
    print(f"Labeled mask:\n{labeled_mask}")

    # Test 3D case
    binary_mask_3d = torch.zeros((7, 7, 7))
    binary_mask_3d[1:6, 1:6, 1:6] = 1  # Solid cube
    binary_mask_3d[2:5, 2:5, 2:5] = 0  # Internal cavity
    binary_mask_3d[3, 3, 3] = 1  # Central point

    labeled_mask_3d = extract_connected_components(binary_mask_3d, orthogonal_hops=1)
    print(f"\n3D labeled mask shape: {labeled_mask_3d.shape}")
    print(f"Number of components: {labeled_mask_3d.max().item()}")
    print(f"Labeled mask:\n{labeled_mask_3d}")
    
    # Test connectivity validation
    try:
        # This should fail - 2D with 3-hops (max is 2 for 2D)
        extract_connected_components(binary_mask_2d, orthogonal_hops=3)
    except ValueError as e:
        print(f"\nExpected error caught: {e}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\nTesting with CUDA...")
        cuda_mask = binary_mask_2d.cuda()
        cuda_labeled = extract_connected_components(cuda_mask, orthogonal_hops=1)
        print(f"CUDA labeled mask shape: {cuda_labeled.shape}")
        print(f"Labeled mask device: {cuda_labeled.device}")
