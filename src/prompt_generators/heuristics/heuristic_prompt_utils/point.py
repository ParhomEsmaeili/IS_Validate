#Here we place the abstracted functions which just perform the implementation according to the input only.

#These funcs should be able to generate errors in instances where there are no longer any remaining voxels for prompt placement
#more specifically relevant to the fine granular interaction mechanisms. 

#NOTE: This is not the same as there being no voxels for correction from the start! 
import os 
from os.path import dirname as up
import sys
sys.path.append(up(up(up(up(up(os.path.abspath(__file__)))))))
from src.prompt_generators.heuristics.spatial_utils.boundary_selection import extract_cavity_boundaries, extract_cavity_boundaries_pytorch
import torch

def uniform_random(binary_mask, n, device):
    '''
    Function which generates n spatial coordinate from the input binary mask, assumed to be HW(D)
    
    Returns a tensor of shape n x n_spatial_dims containing the n spatial coordinates.
    '''
    #Generate tensor of spatial coords: is Ncoords x N_dim
    possible_coords = torch.argwhere(binary_mask)
    if not possible_coords.get_device() == device:
        raise Exception('The binary mask must be matching with the input device')
    
    idxs = torch.sort(torch.randint(0, possible_coords.shape[0] - n + 1,(n,), device=device)).values + torch.arange(0, n, device=device)
    coords = possible_coords[idxs, :].to(dtype=torch.int32)
    return coords 

def center():
    raise NotImplementedError('Still need to validate the boundary extraction strategy, then use cdist and find the furthest point from the nearest boundaries for all points in the foreground.')

    #Put exception handling in the instance where the boundary mask is also the error region. In this circumstance it just should
    #pick a random point because there is no center.
