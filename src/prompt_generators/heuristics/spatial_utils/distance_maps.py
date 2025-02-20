#Module intended to contain the functions required for computing distance maps on a binary mask (with respect to the 
#background). 

import os 
from os.path import dirname as up
import sys
sys.path.append(up(up(up(up(up(os.path.abspath(__file__)))))))
from src.prompt_generators.heuristics.spatial_utils.boundary_selection import extract_interior_boundary
from monai.transforms import LabelToContour 

def edt_from_back(binary_mask):
    '''
    Function which generates a distance map of each voxel in the foreground of a binary mask, to the nearest 
    background voxel

    Returns: Distance map from the background, with NaNs in place for the background.
    '''