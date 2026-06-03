import torch
from .bbox_validation import check_bbox_validity
import copy

def generate_jitter(
    sampling_config: dict,
    context_config: dict,
    ):
    '''
    Helper function which generates jitter parameters according to the provided jitter configuration.
    
    Output: Torch tensor of shape (1,6) representing the jitter to apply to the bounding box extrema.

    Required inputs in sampling_config:
    
        Dimensionality: int, either 2 or 3, representing the dimensionality of the bounding box to be jittered.
            
            If dimensionality is 2, then the collapsed dimension must also be specified to ensure that no jitter is applied
            to the collapsed dimension, which would be invalid.

        Jitter_config: dict
            'type': str, either absolute or relative. This indicates whether the parameterisation is an absolute value,
            or a relative value which needs to be converted to an absolute value based on 1) dimensions of the image
            2) dimensions of the bounding box. relative therefore falls under relative_box, relative_image. 
            
            'sampling_mechanism': str. Represents the mechanism for generating the jitter according to 
                'uniform_integer',
            
            'jitter_symmetric': bool. Whether to apply a symmetric jitter, i.e. min_X and max_X get jittered with the same
            "vector" (size and direction), or whether we will sample distinct jitter values. 
            
        Jitter_parameterisation: list 
            contains the parameters for jitter generation -> informs the sampling mechanism. E.g., raw upper limits,
            or the relative S.F.
    
    context_config: A dictionary containing sample-level/prompt-level specific context
        
        sampling_dimensions: torch.Size representing the dimensions of the array from which the bbox was generated. 
            Always length 3, for 2D bboxes we will require the collapsed dimension to be of size 0. 
            
        bbox_extrema: torch.Tensor representing the extrema of the bounding box. This may be required for specific types 
        of jitter mechanisms.

        collapsed_dim: int | None. If the bbox is 2D this indicates which dimension is collapsed.
        expected_dimensionality: int, either 2 or 3, representing the expected dimensionality of the bounding box. This is required to understand how to check the validity of the bbox, e.g., whether to check for degenerate dimensions, and which dimensions to check for non-negativity and bounds.
    '''
    assert context_config['bbox_extrema'].shape == (1, 6), "Bounding box extrema must have shape (1, 6) representing [min_x, min_y, min_z, max_x, max_y, max_z]"
    assert isinstance(context_config['sampling_dimensions'], torch.Size), "Sampling dimensions must be a torch.Size object"
    assert len(context_config['sampling_dimensions']) == 3, "Sampling dimensions must have 3 elements representing the size of the"
    "array in each spatial dimension from which the bbox was generated (even if the array is originally volumetric a "
    "2D bbox would essentially be a 2D array for the sampling region! -> you would not jitter along the collapsed"
    "dimension!)."


    assert sampling_config != None, "Sampling config cannot be None, we need it to generate jitter parameters."

    if 'dimensionality' not in sampling_config:
        raise KeyError("Sampling config must contain the key 'dimensionality' to specify whether the bounding box is 2D or 3D.")
    
    if 'jitter_config' not in sampling_config:
        raise KeyError("Sampling config must contain the key 'jitter_config' to specify the mechanism for generating jitter.")
    else:
        if 'jitter_symmetric' not in sampling_config['jitter_config']:
            raise KeyError("Sampling config must contain the key 'jitter_symmetric' to specify whether to apply symmetric jitter.")
        if 'type' not in sampling_config['jitter_config']: 
            raise KeyError("Sampling config must contain the key 'type' within 'jitter_config' to" \
            "specify whether jitter is absolute or relative, and whether/how it needs to be processed into absolute values.")
        if 'sampling_mechanism' not in sampling_config['jitter_config']:
            raise KeyError("Sampling config must contain the key 'sampling_mechanism' within 'jitter_config' to specify the mechanism for generating jitter.")
    
    if 'jitter_parameterisation' not in sampling_config:
        raise KeyError("Sampling config must contain the key 'jitter_parameterisation' to specify the parameters for jitter generation.")
    
    # Extract locally and convert list to tensor (since configs come from YAML/JSON)
    raw_param_values = sampling_config['jitter_parameterisation']
    if isinstance(raw_param_values, list):
        raw_param_values = torch.tensor(raw_param_values)
    
    if sampling_config['dimensionality']['expected_dimensionality'] not in [2, 3]:
        raise ValueError("Sampling config 'dimensionality' must be either 2 or 3.")
    
    # sampling dimensions are always 3 elements (H, W, D) representing the 3D volume.
    # This is true for both 2D and 3D bboxes since we're working in 3D space
    # The check for 3 elements is already done at the top of the function.
    
    # jitter_parameterisation has the same number of elements as the dimensionality (2 for 2D, 3 for 3D)
    # For 2D bboxes, this will be padded to 3 elements to match the bbox extrema format
    assert raw_param_values.numel() == sampling_config['dimensionality']['expected_dimensionality'], "Jitter parameterisation must have the same number of elements as the dimensionality specified in the sampling config."

    if sampling_config['dimensionality']['expected_dimensionality'] == 2:
        #We need to insert parameterisatin to pad it to a consistent length, and we will pad with a jitter parameterisation
        #of 0, since we do not want to apply any jitter to the collapsed dimension.
        if 'collapsed_dim' not in context_config:
            raise KeyError("Context config must contain the key 'collapsed_dim' to specify which dimension is collapsed for 2D bounding box generation.")
        collapsed_dim = context_config['collapsed_dim']
        
        if collapsed_dim not in [0, 1, 2]:
            raise ValueError("Context config 'collapsed_dim' must be 0, 1, or 2 to specify which dimension is collapsed for 2D bounding box generation.")
        
        if context_config['bbox_extrema'][0, collapsed_dim] != context_config['bbox_extrema'][0, collapsed_dim + 3]:
            raise ValueError("The extrema for the collapsed dimension must be the same, indicating that it is a single slice and therefore collapsed.")
            
        if context_config['sampling_dimensions'][collapsed_dim] != 0:
            raise ValueError("The size of the collapsed dimension in the sampling dimensions must be 0, indicating that it is a single slice and therefore collapsed.")
        if any(context_config['sampling_dimensions'][dim] == 0 for dim in range(3) if dim != collapsed_dim):
            raise ValueError("The size of the non-collapsed dimensions in the sampling dimensions cannot be 0, otherwise no jitter could be applied to the valid dimensions.")
        
        #Pad the 2D parameterisation (2 elements) to 3D (3 elements) by inserting 0
        #at the collapsed dimension position.
        raw_param_padded = copy.deepcopy(raw_param_values)
        jitter_parameterisation = torch.zeros(3)
        non_collapsed_dims = [d for d in range(3) if d != collapsed_dim]
        for i, dim in enumerate(non_collapsed_dims):
            jitter_parameterisation[dim] = raw_param_padded[i]
        
        #We will assert that the jitter parameterisation for the collapsed dimension is 0.
        if jitter_parameterisation[collapsed_dim] != 0:
            raise ValueError("The jitter parameterisation for the collapsed dimension must be 0 to ensure that no jitter is applied to the collapsed dimension, which would be invalid.")
        if any(jitter_parameterisation[dim] == 0 for dim in non_collapsed_dims):
            raise ValueError("The jitter parameterisation for the non-collapsed dimensions cannot be 0, otherwise no jitter would be applied to the valid dimensions.")
    else:
        # Validate that no dimension is collapsed for 3D bboxes
        min_vals = context_config['bbox_extrema'][0, :3]
        max_vals = context_config['bbox_extrema'][0, 3:]
        collapsed_dims = (min_vals == max_vals).nonzero(as_tuple=True)[0]
        if len(collapsed_dims) > 0:
            raise ValueError(f"3D bounding box extrema must not have collapsed dimensions, but dimensions {collapsed_dims.tolist()} have equal min and max extrema. For 2D bboxes, set expected_dimensionality to 2 and specify collapsed_dim.")
        jitter_parameterisation = copy.deepcopy(raw_param_values) #Being explicit.

    assert jitter_parameterisation.shape == (3,), "After processing, jitter parameterisation must have 3 elements to match the number of dimensions in the bbox extrema"

    #We will assert that the jitter_parameterisation is >=0, negative values are not supported.
    if (jitter_parameterisation < 0).any():
        raise ValueError("Jitter parameterisation values must be non-negative.")

    #For relative types, parameterisation values must be in [0, 1] (fraction of box/array size).
    if sampling_config['jitter_config']['type'] in ('relative_box', 'relative_array'):
        if (raw_param_values < 0).any() or \
           (raw_param_values > 1).any():
            raise ValueError("Relative jitter parameterisation values must be in [0, 1].")

    if sampling_config['jitter_config']['type'] == 'absolute':
        sampling_thresholds = jitter_parameterisation 
    elif sampling_config['jitter_config']['type'] == 'relative_box':
        assert context_config['bbox_extrema'].shape == (1, 6), "Context config 'bbox_extrema' must have shape (1, 6) to use 'relative_box' jitter config type."
        sampling_thresholds = jitter_parameterisation * (context_config['bbox_extrema'][0,3:] - context_config['bbox_extrema'][0,:3]) 
        #We take the size of the box in each dimension and multiply by the relative jitter parameterisation to get 
        #the absolute jitter thresholds.
        
        #NOTE: We do NOT denote the bbox size as being the difference between max and min + 1, as it would
        #only be a single unit of length when we account for the bbox coordinate being placed at the centre of the
        #voxel! Moreover, if we did  + 1 it would result in the bbox being size 1 in a dimension where it was collapsed
        #which is also not what we want! 

    elif sampling_config['jitter_config']['type'] == 'relative_array':
        sampling_thresholds = jitter_parameterisation * torch.tensor(context_config['sampling_dimensions'])
        #We take the size of the array in each dimension and multiply by the relative jitter parameterisation
        #to get the absolute jitter thresholds.
    else:
        raise ValueError("Unsupported jitter config type. Supported types are 'absolute', 'relative_box', and 'relative_array'.")
    
    #We will assert that the sampling thresholds are shape (3,) 
    assert sampling_thresholds.shape == (3,), "Sampling thresholds must have shape (3,) representing the jitter threshold for each dimension."

    #Now that we have the sampling thresholds, we need to generate the jitter. 
    if 'jitter_symmetric' not in sampling_config['jitter_config']:
        raise KeyError("Sampling config must contain the key 'jitter_symmetric' within 'jitter_config' to specify whether to apply symmetric jitter.")
    
    if sampling_config['jitter_config']['jitter_symmetric']:
        if sampling_config['jitter_config']['sampling_mechanism'] == 'uniform_integer':
            #If we use a symmetric jitter, then only need to sample one jitter value per dimension.
            uniform_samples = torch.empty(3).uniform_(-1, 1)
            jitter_samples = (uniform_samples * sampling_thresholds).round().int()
            #Why round and int? Well, we need to displace the coordinate by an integer amount in the voxel representation
            #and we choose to round instead of ceil or floor because of the following diagrams.

            #We assume the bbox extrema are situated at the centre of the voxel, so displacing by a non-integ.
            #if we were between N + 0.5 and N + 1, displacement would be (e.g., N=1):
            #|___|___|___|___|
            #  ^      ^ Clearly this falls into the next voxel, so we want to essentially round to 2.
            #If we were between N and N + 0.5, displacement would be (e.g., N=1):
            #|___|___|___|___|
            #  ^    ^ Clearly this falls into this voxel, so we want to essentially round to 1.

            jitter = jitter_samples.repeat(2).unsqueeze(0)  # Same jitter for min and max
        else:
            raise ValueError("Unsupported sampling mechanism for symmetric jitter. Please check selected configuration.")
    else:
        if sampling_config['jitter_config']['sampling_mechanism'] == 'uniform_integer':

            #We do not use a symmetric jitter, so we need to sample two jitter values per dimension, min and max extrema jitter
            uniform_samples = torch.empty(6).uniform_(-1, 1)
            #Hence we will sample uniformly between -1 and 1 and then multiply by the parameterisation! 
            thresholds_expanded = sampling_thresholds.repeat(2)
            jitter = ((uniform_samples * thresholds_expanded).round().int()).unsqueeze(0) 
            #Why round and int? Well, we need to displace the coordinate by an integer amount in the voxel representation
            #and we choose to round instead of ceil or floor because of the following diagrams.

            #We assume the bbox extrema are situated at the centre of the voxel, so displacing by a non-integ.
            #if we were between N + 0.5 and N + 1, displacement would be (e.g., N=1):
            #|___|___|___|___|
            #  ^      ^ Clearly this falls into the next voxel, so we want to essentially round to 2.
            #If we were between N and N + 0.5, displacement would be (e.g., N=1):
            #|___|___|___|___|
            #  ^    ^ Clearly this falls into this voxel, so we want to essentially round to 1.
            
        else:
            raise ValueError("Unsupported sampling mechanism for non-symmetric jitter. Please check selected configuration.")

    assert jitter.shape == (1, 6), "Generated jitter must have shape (1, 6) to match the shape of the bounding box extrema."
    assert jitter.shape == context_config['bbox_extrema'].shape, "Generated jitter must have the same shape as the bounding box extrema to allow for element-wise addition."

    return jitter

def apply_jitter(
    bbox_extrema: torch.Tensor, 
    jitter_parameters: torch.Tensor,
    context_config: dict
    ):
    '''
    Function which applies a jitter to an input bounding box which has already been generated according to the 
    jitter parameters. It will also check whether the jittered bbox will constitute a valid bbox, and provide a 
    fallback mechanism to ensure that the output is always a valid bbox.

    inputs:
    bbox_extrema: A tensor of shape (1, 6) representing the extrema of the bounding box in the format [min_x, min_y, min_z, max_x, max_y, max_z]
    jitter_parameters: A tensor of shape (1, 6) representing the jitter to apply to the bounding box extrema.
    context_config: A dict which contains some information about the context in which the jitter is being applied:
        required fields:

            dimensionality: dict[
            expected_dimensionality: int, either 2 or 3, representing the expected dimensionality of the bounding box. This is required to understand how to check the validity of the bbox, e.g., whether to check for degenerate dimensions, and which dimensions to check for non-negativity and bounds.
            collapsed_dimension: int, if the expected dimensionality is 2D then this field specifies which dimension was collapsed.
            ]
            
            image_dimensions: torch.Size representing the dimensions of the image volume which the bbox will 
            be situated in. Indicates the permitted range of the bbox coordinates (in voxel-space).
                permitted_range = [0, image_dimension_size - 1] for each dimension. 
                
    outputs: A tensor of shape (1, 6) representing the jittered bounding box extrema in the format [min_x, min_y, min_z, max_x, max_y, max_z]
  
    '''
    original_bbox = bbox_extrema.clone() #To have a copy of the original bbox extrema for fallback if needed. We will not modify this in-place.
    bbox_extrema_jitter = bbox_extrema.clone() #To avoid modifying the original bbox extrema in-place, which could cause issues if we need to fallback to it.
    bbox_extrema_jitter += jitter_parameters
    #Now check the validity of the jittered bbox. 
    
    valid_bool, invalid_indices = check_bbox_validity(bbox_extrema_jitter, context_config)
    if valid_bool:
        pass 
    else:
        assert type(invalid_indices) == list, "Invalid dimensions should be returned as a list of dimension indices which are invalid."
        assert len(invalid_indices) >= 2, "If the bbox is invalid, there should be at least one invalid dimension, and so 2 indices at least."
        assert len(invalid_indices) % 2 == 0, "Invalid indices should come in pairs of min and max for the same dimension, so the length of invalid indices should be even."
        assert all(idx in [0, 1, 2, 3, 4, 5] for idx in invalid_indices), "Invalid indices should be between 0 and 5, representing the min and max extrema for each of the three dimensions."
        for idx in range(3):
            if idx in invalid_indices:
                assert idx + 3 in invalid_indices, f"If the min or max extrema for dimension {idx} is invalid, then both should be invalid and so both index {idx} and {idx + 3} should be in the invalid indices list."
        
        indices_to_fallback = invalid_indices #We need to fallback both the min and max extrema for the invalid dimensions.
        #Fallback to the original bbox dimensions for each invalid dimension.
        bbox_extrema_jitter[0, indices_to_fallback] = original_bbox[0, indices_to_fallback]

    output_bbox = bbox_extrema_jitter.clone() #To ensure we are not returning a reference to the same tensor which could be modified elsewhere
    torch.cuda.empty_cache()
    return output_bbox



def jitter_bbox(
    bbox: torch.Tensor,
    jitter_config: dict,
    context_config: dict
    ):
    '''
    Function which wrap the process of generating and applying a jitter to the bbox extrema. Intended to be called within a master function which
    generates either 2D or 3D bbox prompts.

    inputs: 
    
    bbox: torch.Tensor of shape (1, 6) representing the extrema of the bounding box in the format [min_x, min_y, min_z, max_x, max_y, max_z]
    
    jitter_config: A dict containing the desired configuration for the jitter generation process -> just sets the mechanisms and parameterisation for
    generating the jitter. Expected structure:
        {
            'type': str,  # 'absolute', 'relative_box', or 'relative_array'
            'sampling_mechanism': str,  # 'uniform_integer'
            'jitter_symmetric': bool,  # Whether to apply symmetric jitter
            'parameterisation': torch.Tensor,  # Jitter parameterisation values (1D tensor of length 2 for 2D, 3 for 3D)
        }
    
    context_config: A dict which contains some of the information regarding the constraints of the 
    context in which the bbox was generated, and hence the constraints for performing a jitter. 
    
    Required fields:
        -expected_dimensionality: int, either 2 or 3, representing the expected dimensionality of the bounding box. This is required to understand how to check the validity of the bbox, e.g., whether to check for degenerate dimensions, and which dimensions to check for non-negativity and bounds.
        -collapsed_dimension: int, if the expected dimensionality is 2D then this field
            - these will be wrapped^ but cannot be done in the general wrapper as we only want to pull the
            keys from the registry at the level of the master function. 

        - 'bbox_extrema': torch.Tensor of shape (1, 6)
        - 'image_dimensions': torch.Size representing the dimensions of the image volume which the bbox will 
            be situated in. Indicates the permitted range of the bbox coordinates (in voxel-space).
                permitted_range = [0, image_dimension_size - 1] for each dimension. 
                
    '''
    # Append bbox_extrema to context_config for generate_jitter
    # context_config_with_bbox = context_config.copy()
    # context_config_with_bbox['bbox_extrema'] = bbox
    
    if context_config['expected_dimensionality'] == 2:
        assert 'collapsed_dim' in context_config, "Context config must contain 'collapsed_dim' key to specify which dimension is collapsed for 2D bounding box jittering."

    generate_jitter_context_config = {
        'sampling_dimensions': context_config['sampling_dimensions'],
        'bbox_extrema': context_config['bbox_extrema'],
        'collapsed_dim': context_config.get('collapsed_dim', None),
        'expected_dimensionality': context_config['expected_dimensionality']
        #We will set the collapsed_dim to None if it is not provided in the context_config, 
        # since it is only relevant for 2D bboxes and we can handle that case within the 
        # generate_jitter function.
    }
    # Generate jitter
    jitter = generate_jitter(
        jitter_config, 
        generate_jitter_context_config
        )
    
    #Now we will generate the context config for applying the jitter. 
    apply_jitter_context_config = {
        'dimensionality': {
            'expected_dimensionality': context_config['expected_dimensionality'],
            'collapsed_dimension': context_config.get('collapsed_dim', None)
        },
        'image_dimensions': context_config['image_dimensions']
    }
    # Apply jitter
    result = apply_jitter(bbox, jitter, apply_jitter_context_config)
    
    return result