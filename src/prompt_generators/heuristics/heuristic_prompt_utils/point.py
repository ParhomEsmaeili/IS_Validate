#Here we place the abstracted functions which just perform the implementation according to the input only.

#These funcs should be able to generate empty lists in instances where the sampling region has no valid positions.

#NOTE: This is not the same as there being no voxels for correction from the start! 
import os 
from os.path import dirname as up
import sys
import warnings 
sys.path.append(up(up(up(up(up(os.path.abspath(__file__)))))))
from src.prompt_generators.heuristics.spatial_utils.distance_maps import edt_from_back
import torch
import gc

def uniform_random(binary_mask: torch.Tensor, args: dict):
    '''
    Function which generates n_max spatial coordinate from the input binary mask, assumed to be HW(D)

    Returns a list of length n_max (if possible) with tensors denoting spatial coords with shape 1 x n_spatial_dims. If n_max > num possible (which is > 0) then returns
    a list of num_possible. If num_possible = 0. Then it returns an empty list.
    '''
    #Extract n_max from args. It should be the only argument in args for this function. We use a dictionary in order to enable flexibility
    #for abstracting the process of extracting heuristic functions when constructing the prompt generation object.

    required_args = {"n_max"} 
    if (set(args.keys()) & required_args) - (set(args.keys()) | required_args):
        raise KeyError(f"Disparity in the required args and provided args for uniform random point sampling function. Do not overload \n"
                       "the function with additional parameters.")
    n_max = args.get("n_max")
    device = binary_mask.device
    #Generate tensor of spatial coords: is Ncoords x N_dim
    
    possible_coords = torch.argwhere(binary_mask.to(device=device, dtype=torch.int32))
    #NOTE: We use int32 here because it is more than enough to represent the coordinates in the range of the input binary mask.
    
    if possible_coords.shape[0] >= n_max:
        #If there are sufficient voxels, return N
        idxs = torch.sort(torch.randint(0, possible_coords.shape[0] - n_max + 1,(n_max,), device=device)).values + torch.arange(0, n_max, device=device)
        #Yes, this line is a bit overloaded.. we didn't want to use randperm because of time complexity. We sample indices
        #with possible repetition and then offset them to ensure uniqueness. The upper limit is set to ensure the indices stay
        # within bounds after the offset.
        coords = possible_coords[idxs, :].clone().to(dtype=torch.int32) 
        #We can use int32 because it would be more than enough to represent the coordinates in the range of the input binary mask.
        
        #NOTE: Don't think the garbage collector is actually doing anything here, so will be commenting it out as it is slowing down
        # del possible_coords
        # gc.collect() 
        torch.cuda.empty_cache()
        return list(coords.split(1, 0))
    elif possible_coords.shape[0] < n_max and possible_coords.shape[0] != 0:
        #If there are not sufficient voxels greater than the upper limit provided, return the max quantity which is all of them.
        #idxs = torch.sort(torch.randint(0, 1,(possible_coords.shape[0],), device=device)).values + torch.arange(0, possible_coords.shape[0], device=device)
        coords = possible_coords.to(dtype=torch.int32)
        #NOTE: Don't think the garbage collector is actually doing anything here, so will be commenting it out as it is slowing down
        # del possible_coords
        # gc.collect() 
        torch.cuda.empty_cache()
        return list(coords.split(1, 0)) 
    elif possible_coords.shape[0] == 0:
        #NOTE: Don't think the garbage collector is actually doing anything here, so will be commenting it out as it is slowing down
        # del possible_coords
        # gc.collect() 
        torch.cuda.empty_cache()
        return []

def center():

    raise NotImplementedError('Still cannot be used as it must occur on a component by component basis!')
    raise NotImplementedError('Still need to validate the boundary extraction strategy?')
    
    #TODO: How do we define the center.... currently was using a definition that center = point farthest from background.
    #Centre of mass-like approaches probably wouldn't work as the centroid could fall outside of a gt.
    #NOTE: need to give some thought for how to compute, we could easily just use the numpy based method for extracting distance map too but would need containerisation first. 

    #Put exception handling in the instance where the boundary mask is also the error region. In this circumstance it just should
    #pick a random point because there is no center.
