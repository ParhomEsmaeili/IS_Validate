#Module containing the checkers for the pseudo-ui class.
from typing import Union
import torch 
from monai.data import MetaTensor 


def check_empty(tensor: MetaTensor, bg_val):
    '''
    Basic function which checks whether an input tensor has any foreground voxel. This is determined according to
    the bg val provided.. 
    '''
    if not isinstance(tensor, MetaTensor):
        raise TypeError('The input tensor can only be a Monai metatensor')
    
    return torch.all(tensor == bg_val)

def check_semantic_id_dict(semantic_id_dict: dict):
    '''
    Basic function which implements some checks on the semantic id dictionary.
    '''
    if not isinstance(semantic_id_dict, dict) or not semantic_id_dict:
        raise TypeError('Semantic id-code mapping must be provided as a non-empty dict.')
        
    def check_ints(semantic_id_dict: dict):
        if not all(isinstance(k, str) for k in semantic_id_dict.keys()):
            raise TypeError("All keys in semantic_id_dict must be strs")
        if not all(isinstance(k, int) for k in semantic_id_dict.values()):
            raise TypeError("All the values in semantic_id_dict must be ints")
        
    def check_bg(semantic_id_dict: dict):
        if 'background' not in semantic_id_dict.keys():
            raise KeyError('The background class must be in the semantic id dict.')
        else:
            if semantic_id_dict['background'] != 0:
                raise ValueError('The value of the background class must be zero.')
    
    def check_fg_empty(semantic_id_dict):
        if not set(semantic_id_dict.keys()).difference({'background'}): #Empty set evaluates as False.
            raise Exception('There must be at least one foreground class!')
    

    check_ints(semantic_id_dict)
    check_bg(semantic_id_dict)
    check_fg_empty(semantic_id_dict)


