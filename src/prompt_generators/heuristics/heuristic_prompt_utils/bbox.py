#Intended for the abstract functions utilised in generating bbox prompts.
import copy
import os 
from os.path import dirname as up
import sys
import torch 
from typing import List
sys.path.append(up(up(up(up(up(os.path.abspath(__file__)))))))
from src.prompt_generators.heuristics.spatial_utils.component_extraction import (
    two_d_components_generation, 
    three_d_components_generation,
    select_component
)
from src.prompt_generators.heuristics.spatial_utils.spatial_extent import has_min_spatial_extent_2d
from src.prompt_generators.heuristics.heuristic_prompt_utils.bbox_utils.bbox_validation import check_bbox_validity
from src.prompt_generators.heuristics.heuristic_prompt_utils.bbox_utils.bbox_augmentations import jitter_bbox

def bbox_from_binary_mask(
    binary_mask: torch.Tensor, 
    args: dict
    ):
    '''
    Unified function to generate bounding boxes from binary masks.
    Handles 2D, 3D, and 2.5D based on the dimensionality argument.
    
    This function:
    1. Structures configs necessary for component extraction
    2. Extracts components from the binary mask
    3. Handles cases where sampling a bbox is not compatible
    4. Samples a bbox from the component mask if sampling a bbox is possible.
    5. Applies any augmentation utils (jittering, etc.)
    
    Args:
        binary_mask: A binary mask tensor of shape (H,W,D)
        args: Configuration dictionary with required keys:
            - 'dimensionality': 2, 3, or '2.5D'
            - 'collapsed_dim': int (0, 1, or 2) - required for 2D/2.5D bbox generation
            - 'component_sampling_config': dict (not optional)
            - 'augmentation_config': dict (optional)
    
    Returns:
        bbox_list: List of torch.Tensor, each of shape (1, 6) representing a 2D/3D bbox.
                   Empty list when no bbox could be generated.

    Raises:
        KeyError: If required configuration keys are missing
    '''
    # Validate input
    if binary_mask.dim() != 3:
        raise ValueError("Input binary_mask must be 3D for bbox generation")
    
    if binary_mask.sum() == 0:
        return []
    
    # Validate args
    if args is None:
        raise ValueError("args cannot be None")

    # Extract sampling region (component) from binary mask
    component_sampling_config = args.get('component_sampling_config')
    if component_sampling_config is None:
        raise ValueError("args must contain the key 'component_sampling_config' to specify the configuration for component extraction from the binary mask.")
    #Now lets assert that we have the necessary configs.
    if 'dimensionality' not in component_sampling_config:
        raise KeyError("args must contain the key 'dimensionality' to specify whether the bounding box is 2D or 3D")
    if component_sampling_config['dimensionality'] not in [2, 3, '2.5D']:
        raise ValueError("component_sampling_config['dimensionality'] must be 2, 3, or '2.5D'")


    #Now lets assert that the dimensionality in the component sampling config matches the dimensionality
    #in the args generally.
    if 'dimensionality' not in args:
        raise KeyError("args must contain the key 'dimensionality' to specify whether the bounding box is 2D or 3D")
    else:
        assert args['dimensionality'] == component_sampling_config['dimensionality'], "Dimensionality specified in args must match the dimensionality specified in component_sampling_config"
    
    #If dimensionality is 2D or 2.5D, then lets assert that the collapsed dim is also specified, and identical.
    if component_sampling_config['dimensionality'] in [2, '2.5D']:
        if 'collapsed_dim' not in component_sampling_config:
            raise KeyError("args must contain the key 'collapsed_dim' to specify which dimension is collapsed for 2D/2.5D bounding box generation")
        if 'collapsed_dim' not in args:
            raise KeyError("args must contain the key 'collapsed_dim' to specify which dimension is collapsed for 2D/2.5D bounding box generation")

        if args['collapsed_dim'] != component_sampling_config['collapsed_dim']:
            raise ValueError("The collapsed dimension specified in args must match the collapsed dimension specified in component_sampling_config for 2D/2.5D bounding box generation.")

    if component_sampling_config['dimensionality'] in [2, '2.5D']:
        if 'collapsed_dim' not in component_sampling_config:
            raise KeyError("args must contain the key 'collapsed_dim' for 2D/2.5D bbox generation")
        if component_sampling_config['collapsed_dim'] not in [0, 1, 2]:
            raise ValueError("component_sampling_config['collapsed_dim'] must be 0, 1, or 2")
    if 'region_extraction_config' not in component_sampling_config:
        raise KeyError("args must contain the key 'region_extraction_config' to specify the configuration for region extraction from the binary mask.")
    else:
        if 'connectivity' not in component_sampling_config['region_extraction_config']:
            raise KeyError("component_sampling_config['region_extraction_config'] must contain the key 'connectivity' to specify the connectivity for region extraction.")
        else:
            if component_sampling_config['region_extraction_config']['connectivity'] not in [1, 2, 3]:
                raise ValueError("component_sampling_config['region_extraction_config']['connectivity'] must be 1, 2, or 3.")
        
        if 'component_selection_process' not in component_sampling_config['region_extraction_config']:
            raise KeyError("component_sampling_config['region_extraction_config'] must contain the key 'component_selection_process' to specify the process for selecting components from the binary mask.")
        else:
            #If it does, we need to assert that whatever the process is also has the parameterisation passed through.
            process = component_sampling_config['region_extraction_config']['component_selection_process']
            if process not in component_sampling_config['region_extraction_config']:
                raise KeyError(f"component_sampling_config['region_extraction_config'] must contain the key '{process}' to specify the parameterisation for the selected component selection process.")

        #Now some specifics for 2d bbox generation.
        if component_sampling_config['dimensionality'] == 2:
            if 'slice_selection' not in component_sampling_config['region_extraction_config']:
                raise KeyError("For 2D bbox generation, component_sampling_config['region_extraction_config'] must contain the key 'slice_selection' to specify the process for selecting the slice from which to generate the bbox.")
        elif component_sampling_config['dimensionality'] == '2.5D':
            if 'slice_selection' in component_sampling_config['region_extraction_config']:
                raise ValueError("For 2.5D bbox generation, 'slice_selection' should not be specified — all slices along the collapsed dimension are iterated.")

    if binary_mask.sum() == 0 and binary_mask.unique().numel() == 1:
        torch.cuda.empty_cache()
        return []

    # For 2.5D, delegate to two_point_five_d_bbox_from_binary_mask
    if args['dimensionality'] == '2.5D':
        return two_point_five_d_bbox_from_binary_mask(binary_mask, args)

    # Extract sampling region
    is_compatible, component_mask, slice_idx = extract_sampling_region(binary_mask, component_sampling_config)

    if not is_compatible:
        torch.cuda.empty_cache()
        return []
    else:
        # Sanity check: extract_sampling_region guarantees non-zero components for is_compatible=True,
        # so an all-zero mask here indicates a logic error in component extraction.
        if component_mask.sum() == 0:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "extract_sampling_region returned is_compatible=True but "
                "component_mask is all zeros — this indicates a logic error "
                "in component extraction."
            )

    #Now lets generate a bbox from the binary mask.
    bbox_args = {
            'dimensionality': args['dimensionality']
        }
    if args['dimensionality'] == 2:
        bbox_args.update({
            'collapsed_dim': args['collapsed_dim'],
            'collapsed_slice_idx': slice_idx 
        })
        
    bbox = bbox_extrema(component_mask.bool(), bbox_args)

    # Check bbox validity with critical_failure=True for initial extraction
    check_validity_context_config = {
        'dimensionality': {
            'expected_dimensionality': args['dimensionality'],
            'collapsed_dimension': args['collapsed_dim'] if args['dimensionality'] == 2 else None
        },
        'image_dimensions': binary_mask.shape
    }
    valid_bool, _ = check_bbox_validity(bbox, check_validity_context_config, critical_failure=True)
    if not valid_bool:
        raise Exception('Somehow critical failure is set to True, but a failure occured and was not flagged explicitly')
    
        ######### Now lets apply any augmentations if configured to do so #############    
    augmentation_config = args.get('augmentation_config')
    #This one may be Nonetype if no augmentation is desired.
    
    if augmentation_config == None:
        #If the augmentation config is nonetype, then we have no augmentation!
        torch.cuda.empty_cache()
        return [bbox]
    else:
        ####### Now we generate the augmentation inputs, according to what is required by the 
        # augmentation util ##########
        
        #First we will check that each augmentation in the augmentation config is supported.
        supported_augmentations = [
            'jitter'
        ]
        if any(augmentation not in supported_augmentations for augmentation in augmentation_config):
            raise ValueError(f"One of the augmentations is not supported, currently only the following"
            f"augmentations are supported: {supported_augmentations}. Please check the augmentation config.")
        
        #We have a registry of variables which we will use to generate a dictionary of the context which
        #we want to actually use.
        
        augmentation_context_config_registry = {
            'image_dimensions': binary_mask.shape,
            'sampling_dimensions': binary_mask.shape if args['dimensionality'] == 3 else torch.Size([binary_mask.shape[i] if i != args['collapsed_dim'] else 0 for i in range(3)]),
            #This is the dimensions of the array from which the bbox was generated,
            #NOTE: If it was 2D then the collapsed dim must have size 0 here, we realistically cannot have
            #a relative augmentation based on a dimension which wouldn't have factored into a 2D bbox
            #placement.
            'bbox_extrema': bbox.clone(),
            #This is the extrema of the bbox before augmentation, which may be required for certain types of augmentation.
            'collapsed_dim': args['collapsed_dim'] if args['dimensionality'] == 2 else None,
            'expected_dimensionality': args['dimensionality']
        }

        augmentation_context_config = {
            augmentation: {
            key: augmentation_context_config_registry[key] for key in augmentation_config[augmentation]['context_parameters']
        }
            for augmentation in augmentation_config
        }

        
        #Now we can call the augmentation util to apply the desired augmentation to the bbox.
        #We will iterate through the augmentations in the config and apply them sequentially.
        for augmentation in augmentation_config:
            if augmentation == 'jitter':
                bbox = jitter_bbox(
                    bbox, 
                    augmentation_config[augmentation], 
                    augmentation_context_config[augmentation]
                )
            else:
                raise ValueError(f"Unsupported augmentation {augmentation}. Please check the augmentation config.")
        torch.cuda.empty_cache()
        return [bbox]


def two_point_five_d_bbox_from_binary_mask(
    binary_mask: torch.Tensor,
    args: dict
) -> List[torch.Tensor]:
    """
    Generate a list of 2D bounding boxes, one per valid slice of a 3D component.

    This function:
    1. Extracts 3D connected components (same as the 3D bbox path)
    2. Selects the top-k components
    3. Iterates over each slice along the collapsed dimension
    4. For each slice with sufficient spatial extent, generates a 2D bbox
    5. Applies per-slice augmentation if configured

    Args:
        binary_mask: A binary mask tensor of shape (H, W, D)
        args: Configuration dictionary with required keys:
            - 'dimensionality': '2.5D'
            - 'collapsed_dim': int (0, 1, or 2)
            - 'component_sampling_config': dict
            - 'augmentation_config': dict (optional)

    Returns:
        bbox_list: List of torch.Tensor, each of shape (1, 6) representing a 2D bbox.
                   Empty list if no bboxes could be generated.
    """
    component_sampling_config = args['component_sampling_config']
    region_config = component_sampling_config['region_extraction_config']
    connectivity = region_config['connectivity']
    collapsed_dim = args['collapsed_dim']
    min_length = region_config.get('min_length', 2)

    # Guard: empty mask
    if binary_mask.sum() == 0:
        return []

    # Step 1: Extract 3D components
    components_3d, is_compatible = three_d_components_generation(
        binary_mask, connectivity, min_length
    )
    if not is_compatible:
        return []

    # Step 2: Select the component
    component_selection_config = {
        'component_selection_process': copy.deepcopy(region_config['component_selection_process'])
    }
    component_selection_config.update({
        component_selection_config['component_selection_process']:
            copy.deepcopy(region_config[component_selection_config['component_selection_process']])
    })
    selected_3d = select_component(components_3d, component_selection_config)

    if selected_3d.sum() == 0:
        return []

    # Step 3: Iterate slices along collapsed_dim
    valid_bboxes = []
    num_slices = binary_mask.shape[collapsed_dim]

    for s in range(num_slices):
        try:
            # Extract 2D slice from the selected component
            slice_2d = selected_3d[tuple(s if i == collapsed_dim else slice(None) for i in range(3))]

            # Check spatial extent — skip slices that are too thin
            if min_length >= 2 and not has_min_spatial_extent_2d(slice_2d, min_length):
                continue

            # Re-insert the 2D slice into a 3D volume (bbox_extrema expects 3D input)
            slice_3d = torch.zeros_like(binary_mask)
            slice_3d[tuple(s if i == collapsed_dim else slice(None) for i in range(3))] = slice_2d

            # Compute 2D bbox from this slice
            bbox_args = {
                'dimensionality': 2,
                'collapsed_dim': collapsed_dim,
                'collapsed_slice_idx': s
            }
            slice_bbox = bbox_extrema(slice_3d.bool(), bbox_args)

            # Apply per-slice jitter if configured
            augmentation_config = args.get('augmentation_config')
            if augmentation_config is not None:
                augmentation_context_config_registry = {
                    'image_dimensions': binary_mask.shape,
                    'sampling_dimensions': torch.Size([
                        binary_mask.shape[i] if i != collapsed_dim else 0 for i in range(3)
                    ]),
                    'bbox_extrema': slice_bbox.clone(),
                    'collapsed_dim': collapsed_dim,
                    'expected_dimensionality': 2
                }

                augmentation_context_config = {
                    aug: {
                        key: augmentation_context_config_registry[key]
                        for key in augmentation_config[aug]['context_parameters']
                    }
                    for aug in augmentation_config
                }

                for augmentation in augmentation_config:
                    if augmentation == 'jitter':
                        slice_bbox = jitter_bbox(
                            slice_bbox,
                            augmentation_config[augmentation],
                            augmentation_context_config[augmentation]
                        )

            valid_bboxes.append(slice_bbox)
        except (ValueError, RuntimeError):
            # Skip slices where bbox generation or augmentation fails
            continue

    if len(valid_bboxes) == 0:
        torch.cuda.empty_cache()
        return []

    torch.cuda.empty_cache()
    return valid_bboxes


def bbox_extrema(
    binary_mask: torch.Tensor,
    bbox_args: dict,
    ):
    '''
    Helper function which computes the bounding box extrema from a binary mask. 

    inputs: 
        binary_mask: A binary mask tensor, can have shape (D, H, W) or (H, W) depending on whether 3D or 2D.
        bbox_args: A dict which contains information about the context in which we are generating
        the bbox. Required fields:
            
            dimensionality: int, either 2 or 3, representing the dimensionality of the bounding box
            desired.
                If dimensionality is 2, then the following additional fields are required:
            
                    collapsed_dim: int, the dimension to collapse for 2D bounding box generation (0, 1, or 2)
            
                    collapsed_slice_idx: int, the slice index to use for the collapsed dimension in 2D bounding box generation.
                    This will be used to cross-reference with the extrema calculated from the binary mask to enforce
                    consistency.

    outputs: 
        bbox: A tensor of shape (1, 6) representing the bounding box extrema in the format [min_x, min_y, min_z, max_x, max_y, max_z]

    '''
    if binary_mask.dtype != torch.bool:
        raise TypeError("Input binary mask must be of boolean type.")
    
    assert bbox_args != None, "Bbox_args cannot be None, we need some context parameters, e.g., which dimension is being collapsed'"
    if 'dimensionality' not in bbox_args:
        raise KeyError("Bbox_args must contain the key 'dimensionality' to specify whether the bounding box is 2D or 3D.")
    if bbox_args['dimensionality'] not in [2, 3]:
        raise ValueError("Bbox_args 'dimensionality' must be either 2 or 3.")
    if bbox_args['dimensionality'] == 2:
        if 'collapsed_dim' not in bbox_args:
            raise KeyError("Bbox_args must contain the key 'collapsed_dim' to specify which dimension to collapse for 2D bounding box generation.")
        if bbox_args['collapsed_dim'] not in [0, 1, 2]:
            raise ValueError("Bbox_args 'collapsed_dim' must be 0, 1, or 2 to specify which dimension to collapse for 2D bounding box generation.")
        if 'collapsed_slice_idx' not in bbox_args:
            raise KeyError("Bbox_args must contain the key 'collapsed_slice_idx' to specify the slice index to use for the collapsed dimension in 2D bounding box generation.")
        
    dims = binary_mask.dim()
    if dims != 3:
        raise ValueError("Input binary mask must be either 2D or 3D.")
    # if dims != bbox_args['dimensionality']:
    #     raise ValueError("Dimensionality of the input binary mask does not match the specified 'dimensionality' in bbox_args.")
    
    #We will use i,j,k convention to make it clearer!
    positions = torch.nonzero(binary_mask, as_tuple=False)
    #positions creates an N x 3 tensor where N is the number of non-zero voxels, each row is the i,j,k coordinate
    #for the non-zero voxels. we then take the min and max along the 0th dimension to get the extrema.
    min_i, min_j, min_k = [x.item() for x in torch.min(positions, dim=0)[0]]
    max_i, max_j, max_k = [x.item() for x in torch.max(positions, dim=0)[0]]
    bbox = torch.tensor([[min_i, min_j, min_k, max_i, max_j, max_k]], device=binary_mask.device)
    if bbox_args['dimensionality'] == 3:
        #3D case: We check if any min = max because this is a failure! We should have not reached this
        #position.
        if any(bbox[0, :3] == bbox[0, 3:]):
            raise ValueError("Invalid bounding box: min and max coordinates are the same for at least one dimension, indicating a degenerate bounding box.")
    elif bbox_args['dimensionality'] == 2:
        #2D case:
        #We will assert three things 1) the extrema on the collapsed dim can match
        # 2) the extrema on the remaining dims cannot match, as this would be a degenerate bbox.
        # 3) The value on the collapsed dimension must match the slice index. 
        if bbox[0, bbox_args['collapsed_dim']] != bbox[0, bbox_args['collapsed_dim'] + 3]:
            raise ValueError("Invalid bounding box: for a 2D bounding box, the extrema on the collapsed dimension must match, but the collapsed dim had extrema of {}".format([bbox[0, bbox_args['collapsed_dim']], bbox[0, bbox_args['collapsed_dim'] + 3]]))
        if bbox[0, bbox_args['collapsed_dim']] != bbox_args['collapsed_slice_idx']:
            raise ValueError("Invalid bounding box: for a 2D bounding box, the extrema on the collapsed dimension must match the specified slice index in bbox_args, but got extrema value {} and expected slice index {}".format(bbox[0, bbox_args['collapsed_dim']].item(), bbox_args['collapsed_slice_idx']))
        if any(bbox[0, [dim for dim in range(3) if dim != bbox_args['collapsed_dim']]] == bbox[0, [dim + 3 for dim in range(3) if dim != bbox_args['collapsed_dim']]]):
            raise ValueError("Invalid bounding box: for a 2D bounding box, the extrema on the non-collapsed dimensions cannot match, as this would indicate a degenerate bounding box or that the bounding box was malformed and placed on the wrong axis, but the following non-collapsed dimensions have matching extrema: {}".format([dim for dim in range(3) if dim != bbox_args['collapsed_dim'] and bbox[0, dim] == bbox[0, dim + 3]]))
    
    assert bbox.numel() == 6
    torch.cuda.empty_cache()
    return bbox

def extract_sampling_region(
    binary_mask: torch.Tensor,
    sampling_config: dict
) -> torch.Tensor:
    '''
    Function which extracts a sampling region from a binary mask based on connected component analysis.
    
    For 2D bbox: Samples a slice according to slice_selection config, calls extract_connected_components
    for 2D connected component analysis, selects the components according to the selection process, and returns the component
    re-inserted into the full volume at the selected slice.
    
    For 3D bbox: Calls extract_connected_components for 3D connected component analysis on the whole
    binary mask, selects the components according to the selection process, and returns the component re-inserted into the
    full volume.
    
    Args:
        binary_mask: A binary mask tensor of shape (D, H, W).
        sampling_config: A dictionary containing sampling configuration with the following structure:
            {
                'dimensionality': int (2 or 3),
                'collapsed_dim': int (0, 1, or 2) - which dimension is collapsed for 2D,
                'region_extraction_config': {
                    'connectivity': int (e.g., 1 for 2D, 3 for 3D),
                    'component_selection_process': str (e.g., 'top-k'),
                    'slice_selection': str (e.g., 'center', 'top', 'bottom', 'random')
                    'component_selection_process_VALUE': parameterisation
                }
            }
    
    Returns:
        is_compatible: Indicating whether the binary mask was compatible with component extraction.
        torch.Tensor: The selected component(s) re-inserted into a volume with the same dimensions as input.
        slice_idx: The index of the slice used for 2D bbox generation, None for 3D, or None if no component.

    Raises:
        KeyError: Required config keys are missing.
        ValueError: Invalid configuration values.
    '''
    
    # Validate dimensionality
    if 'dimensionality' not in sampling_config:
        raise KeyError("sampling_config must contain the key 'dimensionality' to specify whether the bounding box is 2D or 3D.")
    if sampling_config['dimensionality'] not in [2, 3]:
        raise ValueError("sampling_config 'dimensionality' must be either 2 or 3.")
    
    # Validate region_extraction_config
    if 'region_extraction_config' not in sampling_config:
        raise KeyError("sampling_config must contain the key 'region_extraction_config'.")
    
    region_config = sampling_config['region_extraction_config']
    
    if 'connectivity' not in region_config:
        raise KeyError("region_extraction_config must contain the key 'connectivity'.")
    if 'component_selection_process' not in region_config:
        raise KeyError("region_extraction_config must contain the key 'component_selection_process'.")
    
    # Validate slice_selection based on dimensionality
    if sampling_config['dimensionality'] == 2:
        if 'slice_selection' not in region_config:
            raise KeyError("For 2D, region_extraction_config must contain the key 'slice_selection'.")
        if region_config['slice_selection'] not in ['center', 'top', 'bottom', 'random']:
            raise ValueError(f"Invalid slice_selection value: {region_config['slice_selection']}. "
                           f"Must be one of: 'center', 'top', 'bottom', 'random'")
    else:  # 3D
        if region_config.get('slice_selection') is not None:
            raise ValueError("For 3D, slice_selection must be None.")
    
    # Validate collapsed_dim for 2D
    if sampling_config['dimensionality'] == 2:
        if 'collapsed_dim' not in sampling_config:
            raise KeyError("sampling_config must contain the key 'collapsed_dim' for 2D bbox.")
        if sampling_config['collapsed_dim'] not in [0, 1, 2]:
            raise ValueError("sampling_config 'collapsed_dim' must be 0, 1, or 2.")
    
    #We will check that the binary mask even passes the basic requirement that is has some non-zero voxels.
    if binary_mask.sum() == 0 and binary_mask.unique().numel() == 1:
        #If the binary mask is completely empty, then we cannot extract any components, so we will return an empty mask and a compatibility flag of False.
        return False, torch.zeros_like(binary_mask), None


    # Get configuration values
    dimensionality = sampling_config['dimensionality']
    connectivity = region_config['connectivity']
    
    # Get the candidate region based on dimensionality
    if dimensionality == 2:
        # Extract the slice from the binary mask
        collapsed_dim = sampling_config['collapsed_dim']
        slice_selection_config = {
            'slice_selection_strategy': region_config['slice_selection'],
            'collapsed_dim': collapsed_dim
        }
        components, slice_idx, is_compatible = two_d_components_generation(
            binary_mask, 
            slice_selection_config,
            connectivity
            )
        assert components.dim() == 2, "Candidate region should be 2D after slice selection for 2D bbox generation."
    else:  # 3D
        slice_idx = None  # Not applicable for 3D
        # Call extract_connected_components from component_extraction -> this will return a tensor with voxel values 
        # indicating the component ids.
        components, is_compatible = three_d_components_generation(
            binary_mask, 
            connectivity
            )
        
    #If not compatible, then we cant generate any components so need to return an empty mask.
    if not is_compatible:
        return is_compatible, torch.zeros_like(binary_mask), None #Return an empty mask if the input binary mask is not compatible with the selected slice selection strategy, e.g., if there are no non-zero voxels in the selected slice for 2D bbox generation, or if there are no non-zero voxels in the whole volume for 3D bbox generation. This is a design choice to ensure that we always return a valid output, even if it is an empty mask.
    else:
    #Otherwise, if it is, we need to now select the component to use for bbox generation later.
        assert components.sum() > 0, "If the input binary mask is compatible with the selected slice selection strategy, then there should be at least one non-zero voxel in the candidate region for component extraction."
        assert components.unique().numel() > 1, "If the input binary mask is compatible with the selected slice selection strategy, then there should be at least one connected component in the candidate region for component extraction, resulting in more than one unique value (including background) in the components tensor."

        #Now we will perform component selection to obtain the extraction region.
        component_selection_config = {
            'component_selection_process': copy.deepcopy(region_config['component_selection_process'])
        }
        assert component_selection_config['component_selection_process'] in region_config.keys(), (
            "The parameterisation of the component selection mechanism must be specified in the region config "
            "under the key outlined by the selection process")
        
        component_selection_config.update({
            component_selection_config['component_selection_process']: copy.deepcopy(region_config[component_selection_config['component_selection_process']])
        })
        

        selected_components = select_component(
            components,
            component_selection_config
        ) 
        
        # Create output mask with same dimensions as binary_mask
        output_mask = torch.zeros_like(binary_mask)
        
        # insert selected components into output mask
        if dimensionality == 2:
            # For 2D, insert into the selected slice
            assert slice_idx != None, "Slice index should not be None for 2D bounding box generation."
            assert slice_idx >= 0 and slice_idx <= binary_mask.shape[sampling_config['collapsed_dim']] - 1, "Slice index is out of bounds for the collapsed dimension."
            expected_2d_shape = tuple(binary_mask.shape[i] for i in range(3) if i != sampling_config['collapsed_dim'])
            assert selected_components.shape == expected_2d_shape, "Selected components shape does not match the expected shape for the collapsed slice."
            # if collapsed_dim == 0:
            #     output_mask[slice_idx, :, :] = copy.deepcopy(selected_components)
            # elif collapsed_dim == 1:
            #     output_mask[:, slice_idx, :] = copy.deepcopy(selected_components)
            # elif collapsed_dim == 2:
            #     output_mask[:, :, slice_idx] = copy.deepcopy(selected_components)

            # Generalized insertion for any collapsed dimension
            output_mask[tuple(slice_idx if i == sampling_config['collapsed_dim'] else slice(None) for i in range(3))] = copy.deepcopy(selected_components)
        else:  # 3D
            # For 3D, insert into the full volume
            assert selected_components.shape == binary_mask.shape, "Selected components shape does not match the input binary mask shape for 3D bounding box generation."
            output_mask = copy.deepcopy(selected_components)
        
        torch.cuda.empty_cache()

        return is_compatible, output_mask, slice_idx



