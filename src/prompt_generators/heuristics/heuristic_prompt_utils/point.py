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

def uniform_random(binary_mask, n):
    '''
    Function which generates n spatial coordinate from the input binary mask, assumed to be HW(D)
    
    Returns a list of length n (if possible) with tensors denoting spatial coords with shape 1 x n_spatial_dims. If n > num possible (which is > 0) then returns
    a list of num_possible. If num_possible = 0. Then it returns an empty list.
    '''
    device = binary_mask.device
    #Generate tensor of spatial coords: is Ncoords x N_dim
    
    possible_coords = torch.argwhere(binary_mask).to(device=device)
    
    if possible_coords.shape[0] >= n:
        #If there are sufficient voxels, return N
        idxs = torch.sort(torch.randint(0, possible_coords.shape[0] - n + 1,(n,), device=device)).values + torch.arange(0, n, device=device)
        coords = possible_coords[idxs, :].to(dtype=torch.int64)
        return list(coords.split(1, 0))
    elif possible_coords.shape[0] < n and possible_coords.shape[0] != 0:
        #If there are not sufficient voxels, return the max quantity?
        #idxs = torch.sort(torch.randint(0, 1,(possible_coords.shape[0],), device=device)).values + torch.arange(0, possible_coords.shape[0], device=device)
        coords = possible_coords.to(dtype=torch.int64)
        return list(coords.split(1, 0))
    elif possible_coords.shape[0] == 0:
        return []

def center():

    raise NotImplementedError('Still cannot be used as it must occur on a component by component basis!')
    raise NotImplementedError('Still need to validate the boundary extraction strategy?')
    
    #TODO: How do we define the center.... currently was using a definition that center = point farthest from background.
    #Centre of mass-like approaches probably wouldn't work as the centroid could fall outside of a gt.
    #NOTE: need to give some thought for how to compute, we could easily just use the numpy based method for extracting distance map too but would need containerisation first. 

    #Put exception handling in the instance where the boundary mask is also the error region. In this circumstance it just should
    #pick a random point because there is no center.
